start at `gaussian_point_train.py`

create a `GaussianPointCloudTrainer`; config from either cli args or yaml file
create log/summary & output dirs
create summarywriter

load train & val imgs datasets (`train_dataset, val_dataset`) from config.train_dataset_json_path (via `ImagePoseDataset`)
create `scene` `GaussianPointCloudScene` from config.pointcloud_parquet_path, and apply `cuda()`
create `adaptive_controller` `GaussianPointAdaptiveController` w/ config & scene as args
{
  `iteration_counter` = -1, `config` = config, `maintained_parameters` = scene.maintained_parameters (scene), `input_data` = None
  `density_point_info` is optional, default none: `GaussianPointAdaptiveController.GaussianPointAdaptiveControllerDensifyPointInfo`
  create accumulated tensors for num pixels, num in camera, view space position gradients, position gradients, position gradients norm, all in shape of `maintained_parameters.pointcloud[:, 0]`
  create plotting figure and axis (for histogram)
}


create `rasterisation` `GaussianPointCloudRasterisation` w/ config; set `backward_valid_point_hook` to `adaptive_controller.update,`

create `loss_function` `LossFunction` w/ config
best_psnr_score = 0

start train()
init ti
create `train_data_loader`, `val_data_loader` from corresponding datasets
`train_data_loader_iter = cycle(train_data_loader)` (while true yield next from loader)

create Adam `optimizer` from pcl feats, config lr
create Adam `position_optimizer`from pcl, confngi rl
create `scheduler` (torch optim lr expon sched) for `position_optimizer`

`downsample_factor` from config

`recent_losses` deque, l=100

for `iter` in range(`self.config.num_iterations`):

    the core of the training loop:
        getdownsampling factor; zero grad `optimizer`, `position_optimizer`
    	get next batch of img, pointcloud, camera info; downsample if needed; to cuda; prep as rasterizer input
    	get pred color&depth, pxls valid point count, from rasterisation
    	calculate loss, l1_loss, ssim_loss; backward; step optimizer (and position one)
    {
    	`downsample_factor //=2` if `!iter & config.half_downsample_factor_interval and iter>0 and downsample_factor>1`

    	zero grad `optimizer`, `position_optimizer`

    	get next batch of `image_gt`, `q_pointcloud_camera`, `t_pointcloud_camera` (q,t: pclcams), `camera_info` from `train_data_loader_iter`
    	if downsamplefactor>1: downsample image_gt (pxls) and camera info (w,h,intrinsics)
    	call `_downsample_image_and_camera_info` on `image_gt` and `camera_info` w/ `downsample_factor` if `downsample_factor>1`
    	`image_gt`, pclcams, `camera_info.camera_intrinsics` to `cuda`

    	(... denotes same varname as param)
    	prep rasterizer input `gaussian_point_cloud_rasterisation_input` (`raster_input`) `GaussianPointCloudRasterisation.GaussianPointCloudRasterisationInput`
    	point_cloud=scene. ...,
    	point_cloud_features = scene. ...,
    	point_object_id = scene. ...,
    	point_invalid_mask = scene. ...,
    	camera_info, pclcams, color_max_sh_band=iter

    	get `image_pred`, `image_depth`, `pixel_valid_point_count` from rasterisation(`raster_input`)
    	torch clamp `image_pred` to [0,1]
    	`image_pred` permute(2,0,1): (h,w,3) -> (3,h,w)

    	calc `loss`, `l1_loss`, `ssim_loss` w/ `loss_function(image_pred, image_gt, point_invalid_mask, pcl_features)`
    	loss.backward()
    	step `optimizer`, `position_optimizer`
    	append loss to `recent_losses`

    	step scheduler if at `position_optimizer` step interval
    }

    get `magnitude_grad_viewspace_on_image` from `adaptive_controller.input_data` (if not None)
    plot the gradient histogram of `adaptive_controller.input_data`, value histogram of `scene`, `pixel_valid_point_count` to `writer`

    also important: gaussians refinement w/ adaptive controller
    	call `refinement()` on `adaptive_controller`

    the rest is logging, plotting, debugging (and perform validation at certain iters)
    {
    	plot `image_pred` to `adaptive_controller.figure` and `ax` (if `adaptive_controller.has_plot`)

    	if (`iter % log_loss_interval == 0`):
    		log `loss`, `l1_loss`, `ssim_loss` to `writer` (and console if `print_loss_to_console`)

    	print and clear taichi kernel profiler info at certain intervals

    	write psnr, ssim to `writer` (and console) at log metrics interval

    	record curr iter to be problematic if the loss is more than 1.5 times the average of the last 100 losses

    	add magnitude grad viewspace on image to `writer`; add images to writer, if at log image interval or if the loss is problematic

    	del `image_gt`, `q_pointcloud_camera`, `t_pointcloud_camera`, `camera_info`, `gaussian_point_cloud_rasterisation_input`, `image_pred`, `loss`, `l1_loss`, `ssim_loss`

    	`validation()` at certain intervals or at 5000, 7000 iterations
    }





THE rasterizer (entry: `GaussianPointCloudRasterisation` (nnModule))

class `GaussianPointCloudRasterisation` {
  class `...Config`, `...Input`, `BackwardValidPointHook`

  init {
    create a module function (is `torch.autograd.Function`) w/ forward and backward. basically, the module function is called when rasterizer is called (via forward pass) 
    {
      forward: {
        1. filter points (`filter_point_in_camera`)
        1a. get id bsaed on camera mask; get num pts in cam; allocate mem (gpu)
        2. get 2d feats w `generate_point_attributes_in_camera_plane`
        3. get # of tiles overlapped (for memalloc): `generate_num_overlap_tiles`
        3a. calc presum of number_overlap_tiles; del aftewards
        4. calc key `generate_point_sort_key_by_num_overlap_tiles`
        4a. create lots of tiles, it seems. also `find_tile_start_and_end`
        4b. alloc imgs mem: `rasterized_image`, `...depth`, `pixel_accumulated_alpha`, `pixel_offset_of_last_effectifind_tile_start_and_effective_point`, `pixel_valid_point_count`

        5. render `gaussian_point_rasterisation` 
          and save a lot of vars for backward pass 
      }

      backward: {
        `gaussian_point_rasterisation_backward` is called at some point 
        aswell as `_clear_grad_by_color_max_sh_band`

        something related to deriving spherical harmonics

        finally calls the backward valid point hook
      }
    }
  }

  _clear_grad_by_color_max_sh_band {
    setting certain values in the features to zero, b ecause of maximum spherical harmonics band

  }
}


the `rasterizer's backward valid point hook` is the `adaptive controller's update()`, so we check that out:

class AdaptiveController {
  init {...}

  update { iter counter++
    w/ torch nograd:
      some vars are copied over; some accumulated; some divided 

      if warned up and at certain intervals, call `_find_densify_points` with same args as update()
  }

  _find_densify_points {
    finds points to densify, "it should happened in backward pass before optimiser step.
        so that the original point values are recorded, and when a point is cloned/split, the
        two points are not the same."

    floater and densification 

    alpha b4 sigmoid 
    floaters and transparent points dont overlap 
    find under or overreconstructed pts; dont densify floater,transparents

    ...

    doesnt return anything
  }

  refinement { w/ no grad 
    do nothing if still warming up
    if at densification interval: 
    
    `_add_densify_points`  
    reset accumulated_{num_in_camera, num_pixels, view_space_position_gradients, view_space_position_gradients_avg, position_gradients, position_gradients_norm} tensors to pointcloud[:, 0] size

    reset alpha if at reset interval
  }
}