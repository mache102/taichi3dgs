import argparse
import os
import torch
import numpy as np
from taichi_3d_gaussian_splatting.Camera import CameraInfo
from taichi_3d_gaussian_splatting.GaussianPointCloudRasterisation import GaussianPointCloudRasterisation
from taichi_3d_gaussian_splatting.GaussianPointCloudScene import GaussianPointCloudScene
from taichi_3d_gaussian_splatting.utils import SE3_to_quaternion_and_translation_torch
from PIL import Image
import taichi as ti
from tqdm import tqdm

"""


python headless_render.py --parquet_path_list ./logs/tat_truck_every_8_experiment/best_scene.parquet --num_frames 60  --use_prepared_cameras

"""

def save_image_from_tensor(tensor, filepath):
    """Convert torch tensor to PIL image and save to disk"""
    # Clamp to 0-1 range and convert to CPU
    img = (torch.clamp(tensor, 0, 1) * 255).byte().cpu().numpy()
    # Create PIL image and save
    Image.fromarray(img).save(filepath)
    
class HeadlessRenderer:
    def __init__(self, parquet_path_list, output_dir="renders", 
                 image_height=546, image_width=980, device="cuda",
                 use_prepared_cameras=False):
        # Create output directory
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Configure dimensions
        self.image_height = image_height - image_height % 16
        self.image_width = image_width - image_width % 16
        self.device = device
        self.use_prepared_cameras = use_prepared_cameras
        
        # Load scenes
        scene_list = []
        for parquet_path in parquet_path_list:
            print(f"Loading {parquet_path}")
            scene = GaussianPointCloudScene.from_parquet(
                parquet_path, 
                config=GaussianPointCloudScene.PointCloudSceneConfig(max_num_points_ratio=None)
            )
            scene_list.append(scene)
        
        print("Merging scenes")
        self.scene = self._merge_scenes(scene_list)
        self.scene = self.scene.to(device)
        
        # Set up camera intrinsics - using the prepared intrinsics
        self.camera_intrinsics = torch.tensor(
            [[581.743, 0.0, 490.0], [0.0, 581.743, 273.0], [0.0, 0.0, 1.0]],
            device=device, dtype=torch.float32  # Explicitly set dtype
        )
        
        # Set up prepared camera transform
        self.prepared_camera_transform = torch.tensor(
            [[0.9992602094, -0.0041446825, 0.0382342376, 0.8111615373], 
             [0.0047891027, 0.9998477637, -0.0167783848, 0.4972433596], 
             [-0.0381588759, 0.0169490798, 0.999127935, -3.8378280443], 
             [0.0, 0.0, 0.0, 1.0]],
            device=device, dtype=torch.float32  # Explicitly set dtype
        )
        
        # Set up camera info
        self.camera_info = CameraInfo(
            camera_intrinsics=self.camera_intrinsics,
            camera_width=self.image_width,
            camera_height=self.image_height,
            camera_id=0,
        )
        
        # Set up rasterizer
        self.rasteriser = GaussianPointCloudRasterisation(
            config=GaussianPointCloudRasterisation.GaussianPointCloudRasterisationConfig(
                near_plane=0.8,
                far_plane=1000.,
                depth_to_sort_key_scale=100.
            )
        )
        
    def _merge_scenes(self, scene_list):
        """Merge multiple scenes into one"""
        merged_point_cloud = torch.cat(
            [scene.point_cloud for scene in scene_list], dim=0)
        merged_point_cloud_features = torch.cat(
            [scene.point_cloud_features for scene in scene_list], dim=0)
        
        num_of_points_list = [scene.point_cloud.shape[0] for scene in scene_list]
        start_offset_list = [0] + np.cumsum(num_of_points_list).tolist()[:-1]
        end_offset_list = np.cumsum(num_of_points_list).tolist()
        
        point_object_id = torch.zeros(
            (merged_point_cloud.shape[0],), dtype=torch.int32, device=self.device)
        for idx, (start_offset, end_offset) in enumerate(zip(start_offset_list, end_offset_list)):
            point_object_id[start_offset:end_offset] = idx
            
        merged_scene = GaussianPointCloudScene(
            point_cloud=merged_point_cloud,
            point_cloud_features=merged_point_cloud_features,
            point_object_id=point_object_id,
            config=GaussianPointCloudScene.PointCloudSceneConfig(
                max_num_points_ratio=None
            )
        )
        return merged_scene
        
    def generate_camera_trajectory(self, num_frames=60):
        """Generate a circular camera trajectory around the scene"""
        # Get scene center and radius
        center = self.scene.point_cloud.mean(dim=0)
        radius = torch.norm(self.scene.point_cloud - center, dim=1).max().item() * 1.5
        
        camera_poses = []
        for i in range(num_frames):
            # Calculate angle in radians
            angle = 2 * np.pi * i / num_frames
            
            # Create camera pose matrix (looking at center from different positions on a circle)
            eye = torch.tensor([
                radius * np.sin(angle),
                0.2 * radius,  # Slightly elevated
                radius * np.cos(angle)
            ], device=self.device, dtype=torch.float32) + center
            
            # Create rotation matrix (look at center)
            look_dir = center - eye
            look_dir = look_dir / torch.norm(look_dir)
            
            # Calculate up and right vectors - ensuring same dtype
            up = torch.tensor([0.0, 1.0, 0.0], device=self.device, dtype=torch.float32)
            right = torch.cross(look_dir, up)
            right = right / torch.norm(right)
            up = torch.cross(right, look_dir)
            
            # Create rotation matrix
            rotation = torch.stack([right, up, -look_dir], dim=1)
            
            # Create 4x4 transformation matrix
            transform = torch.eye(4, device=self.device, dtype=torch.float32)
            transform[:3, :3] = rotation
            transform[:3, 3] = eye
            
            camera_poses.append(transform)
            
        return torch.stack(camera_poses, dim=0)
    
    def get_prepared_camera_trajectory(self, num_frames=60):
        """Create variations of the prepared camera pose for a trajectory"""
        base_transform = self.prepared_camera_transform.unsqueeze(0)
        
        # We'll create slight variations of the prepared camera
        # For a simple orbit, we'll rotate around the up axis
        camera_poses = []
        
        # Extract original position
        original_position = base_transform[0, :3, 3]
        
        # Extract original look direction (z-axis of camera)
        original_forward = -base_transform[0, :3, 2]
        original_up = base_transform[0, :3, 1]
        
        # Calculate center point the camera is looking at
        # We'll assume it's some distance along the forward direction
        look_distance = 5.0  # Can be adjusted
        center_point = original_position + look_distance * original_forward
        
        for i in range(num_frames):
            # Calculate angle in radians
            angle = 2 * np.pi * i / num_frames
            
            # Create rotation matrix around world up axis
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            rot_matrix = torch.eye(3, device=self.device, dtype=torch.float32)
            rot_matrix[0, 0] = cos_angle
            rot_matrix[0, 2] = sin_angle
            rot_matrix[2, 0] = -sin_angle
            rot_matrix[2, 2] = cos_angle
            
            # Rotate position around center point
            centered_pos = original_position - center_point
            rotated_pos = torch.matmul(rot_matrix, centered_pos) + center_point
            
            # Create new transform
            transform = base_transform.clone()
            
            # Compute new forward direction (looking at center)
            new_forward = center_point - rotated_pos
            new_forward = new_forward / torch.norm(new_forward)
            
            # Compute new right and up directions
            new_right = torch.cross(new_forward, original_up)
            new_right = new_right / torch.norm(new_right)
            new_up = torch.cross(new_right, new_forward)
            
            # Update rotation part of transform
            transform[0, :3, 0] = new_right
            transform[0, :3, 1] = new_up
            transform[0, :3, 2] = -new_forward  # Negative because camera looks along -z
            
            # Update position
            transform[0, :3, 3] = rotated_pos
            
            camera_poses.append(transform.squeeze(0))
            
        return torch.stack(camera_poses, dim=0)
    
    def render_trajectory(self, camera_poses):
        """Render scene from each camera pose and save to disk"""
        q_pointcloud_camera, t_pointcloud_camera = SE3_to_quaternion_and_translation_torch(camera_poses)
        
        # Create progress bar
        for i, (q, t) in enumerate(tqdm(zip(q_pointcloud_camera, t_pointcloud_camera), 
                                       total=len(q_pointcloud_camera),
                                       desc="Rendering frames")):
            # Render scene
            with torch.no_grad():
                image, depth, _ = self.rasteriser(
                    GaussianPointCloudRasterisation.GaussianPointCloudRasterisationInput(
                        point_cloud=self.scene.point_cloud,
                        point_cloud_features=self.scene.point_cloud_features,
                        point_invalid_mask=self.scene.point_invalid_mask,
                        point_object_id=self.scene.point_object_id,
                        camera_info=self.camera_info,
                        q_pointcloud_camera=q.unsqueeze(0),
                        t_pointcloud_camera=t.unsqueeze(0),
                        color_max_sh_band=3,
                    )
                )
            
            # Save image
            image_path = os.path.join(self.output_dir, f"frame_{i:04d}.png")
            save_image_from_tensor(image, image_path)
            
            # Save depth (optional)
            depth_path = os.path.join(self.output_dir, f"depth_{i:04d}.png")
            depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            save_image_from_tensor(depth_normalized, depth_path)
            
    def render(self, num_frames=60):
        """Generate camera trajectory and render frames"""
        if self.use_prepared_cameras:
            print("Using prepared camera trajectory")
            camera_poses = self.get_prepared_camera_trajectory(num_frames)
        else:
            print("Generating circular camera trajectory")
            camera_poses = self.generate_camera_trajectory(num_frames)
            
        self.render_trajectory(camera_poses)
        print(f"Rendering complete! {num_frames} frames saved to {self.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render 3D Gaussian Splats without GUI")
    parser.add_argument("--parquet_path_list", type=str, nargs="+", required=True, 
                      help="Paths to parquet files containing Gaussian point clouds")
    parser.add_argument("--output_dir", type=str, default="renders",
                      help="Directory to save rendered images")
    parser.add_argument("--num_frames", type=int, default=60,
                      help="Number of frames to render in trajectory")
    parser.add_argument("--image_height", type=int, default=546,
                      help="Height of rendered images")
    parser.add_argument("--image_width", type=int, default=980,
                      help="Width of rendered images")
    parser.add_argument("--use_prepared_cameras", action="store_true",
                      help="Use prepared camera poses instead of generating new ones")
    
    args = parser.parse_args()
    
    # Initialize taichi
    ti.init(arch=ti.cuda, device_memory_GB=4)
    
    # Create renderer and render frames
    renderer = HeadlessRenderer(
        parquet_path_list=args.parquet_path_list,
        output_dir=args.output_dir,
        image_height=args.image_height,
        image_width=args.image_width,
        use_prepared_cameras=args.use_prepared_cameras
    )
    
    renderer.render(num_frames=args.num_frames)