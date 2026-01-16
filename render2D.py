#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import yaml
from os import makedirs
from pathlib import Path
from time import time
from argparse import ArgumentParser, Namespace
import random

import numpy as np
import torch
import torchvision
import open3d as o3d
import typer
from rich.console import Console
from typing_extensions import Annotated
from tqdm import tqdm

from scene import Scene
from gaussian_renderer import render2D, SurfelModel
from arguments import ModelParams, PipelineParams, get_combined_args
from scene.dataset_readers import loadCameras
from utils.general_utils import safe_state
from utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh
from utils.render_utils import generate_path, create_videos

from icecream import ic
# from utils.sfm_utils import save_time

console = Console()


def build_args_from_config(data_cf: dict) -> Namespace:
    """Build an argparse-compatible Namespace from YAML config."""
    render_cf = data_cf.get("render", {})

    args_dict = {
        # Common paths
        "source_path": data_cf.get("source_path", ""),
        "model_path": data_cf.get("model_path", ""),
        "n_views": data_cf.get("n_views", 0),
        # Script-specific args
        "iterations": render_cf.get("iterations", -1),
        "skip_train": render_cf.get("skip_train", False),
        "skip_test": render_cf.get("skip_test", False),
        "skip_mesh": render_cf.get("skip_mesh", False),
        "quiet": render_cf.get("quiet", False),
        "render_path": render_cf.get("render_path", False),
        "voxel_size": render_cf.get("voxel_size", -1.0),
        "depth_trunc": render_cf.get("depth_trunc", -1.0),
        "sdf_trunc": render_cf.get("sdf_trunc", -1.0),
        "num_cluster": render_cf.get("num_cluster", 50),
        "unbounded": render_cf.get("unbounded", False),
        "mesh_res": render_cf.get("mesh_res", 1024),
        "optim_test_pose_iter": render_cf.get("optim_test_pose_iter", 500),
        "infer_video": render_cf.get("infer_video", False),
        "force_debug": render_cf.get("force_debug", False),
        # ModelParams
        "sh_degree": render_cf.get("sh_degree", 3),
        "images": render_cf.get("images", "images"),
        "resolution": render_cf.get("resolution", -1),
        "white_background": render_cf.get("white_background", False),
        "data_device": render_cf.get("data_device", "cuda"),
        "eval": render_cf.get("eval", False),
        "render_items": render_cf.get(
            "render_items", ["RGB", "Alpha", "Normal", "Depth", "Edge", "Curvature"]
        ),
        "init_scale_from_view_depth": render_cf.get(
            "init_scale_from_view_depth", False
        ),
        # PipelineParams
        "convert_SHs_python": render_cf.get("convert_SHs_python", False),
        "compute_cov3D_python": render_cf.get("compute_cov3D_python", False),
        "depth_ratio": render_cf.get("depth_ratio", 0.0),
        "debug": render_cf.get("debug", False),
    }
    return Namespace(**args_dict)


def run_rendering(args, dataset, iteration, pipe):
    """Main rendering logic extracted from the original __main__ block."""
    if getattr(args, "force_debug", False):
        pipe.debug = True
    gaussians = SurfelModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    train_dir = os.path.join(
        args.model_path, "train", "ours_{}".format(scene.loaded_iter)
    )
    test_dir = os.path.join(
        args.model_path, "test", "ours_{}".format(scene.loaded_iter)
    )
    gaussExtractor = GaussianExtractor(gaussians, render2D, pipe, bg_color=bg_color)

    optimized_pose = None
    if args.eval:
        if (not args.skip_test) and (len(scene.getTestCameras()) > 0):
            print("export rendered testing images ...")
            os.makedirs(test_dir, exist_ok=True)
            start_time = time()
            ic(args.optim_test_pose_iter)
            gaussExtractor.reconstruction_optim(
                gaussians,
                scene.getTestCameras(),
                args.optim_test_pose_iter,
                pipe,
                background,
            )
            end_time = time()
            # save_time(dataset.model_path, '[5] render_test', end_time - start_time)
            gaussExtractor.export_image(test_dir)

    else:
        if not args.skip_train:
            print("export training images ...")
            os.makedirs(train_dir, exist_ok=True)
            optimized_pose = np.load(
                Path(args.model_path)
                / "pose"
                / f"ours_{iteration}"
                / "pose_optimized.npy"
            )
            viewpoint_stack = loadCameras(optimized_pose, scene.getTrainCameras())
            gaussExtractor.reconstruction_optim(
                gaussians, viewpoint_stack, 0, pipe, background
            )
            # gaussExtractor.reconstruction(viewpoint_stack)
            gaussExtractor.export_image(train_dir)

        if not args.skip_mesh:
            print("export mesh ...")
            os.makedirs(train_dir, exist_ok=True)
            # set the active_sh to 0 to export only diffuse texture
            gaussExtractor.gaussians.active_sh_degree = 0
            if optimized_pose is None:
                optimized_pose = np.load(
                    Path(args.model_path)
                    / "pose"
                    / f"ours_{iteration}"
                    / "pose_optimized.npy"
                )
            viewpoint_stack = loadCameras(optimized_pose, scene.getTrainCameras())
            gaussExtractor.reconstruction_optim(
                gaussians, viewpoint_stack, 0, pipe, background
            )
            # gaussExtractor.reconstruction(viewpoint_stack)

            # extract the mesh and save
            if args.unbounded:
                name = "fuse_unbounded.ply"
                mesh = gaussExtractor.extract_mesh_unbounded(resolution=args.mesh_res)
            else:
                name = "fuse.ply"
                depth_trunc = (
                    (gaussExtractor.radius * 2.0)
                    if args.depth_trunc < 0
                    else args.depth_trunc
                )
                voxel_size = (
                    (depth_trunc / args.mesh_res)
                    if args.voxel_size < 0
                    else args.voxel_size
                )
                sdf_trunc = 5.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc
                mesh = gaussExtractor.extract_mesh_bounded(
                    voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc
                )

            o3d.io.write_triangle_mesh(os.path.join(train_dir, name), mesh)
            print("mesh saved at {}".format(os.path.join(train_dir, name)))
            # post-process the mesh and save, saving the largest N clusters
            mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
            o3d.io.write_triangle_mesh(
                os.path.join(train_dir, name.replace(".ply", "_post.ply")), mesh_post
            )
            print(
                "mesh post processed saved at {}".format(
                    os.path.join(train_dir, name.replace(".ply", "_post.ply"))
                )
            )

    if args.render_path:
        print("render videos ...")
        traj_dir = os.path.join(
            args.model_path, "traj", "ours_{}".format(scene.loaded_iter)
        )
        os.makedirs(traj_dir, exist_ok=True)
        n_fames = 240
        if optimized_pose is None:
            optimized_pose = np.load(
                Path(args.model_path)
                / "pose"
                / f"ours_{iteration}"
                / "pose_optimized.npy"
            )
        viewpoint_stack = loadCameras(optimized_pose, scene.getTrainCameras())
        cam_traj = generate_path(viewpoint_stack, n_frames=n_fames)
        gaussExtractor.reconstruction(cam_traj)
        gaussExtractor.export_image(traj_dir)
        create_videos(
            base_dir=traj_dir,
            input_dir=traj_dir,
            out_name="render_traj",
            num_frames=n_fames,
        )


def main_typer(
    config_path: Annotated[
        str, typer.Argument(help="Path of the config file")
    ] = "./configurations/barn.cfg",
    eval: Annotated[
        bool, typer.Option("--eval", help="Enable evaluation mode")
    ] = None,
    skip_train: Annotated[
        bool, typer.Option("--skip-train", help="Skip training set rendering")
    ] = None,
    skip_test: Annotated[
        bool, typer.Option("--skip-test", help="Skip test set rendering")
    ] = None,
    skip_mesh: Annotated[
        bool, typer.Option("--skip-mesh", help="Skip mesh extraction")
    ] = None,
    iterations: Annotated[
        int, typer.Option("--iterations", "-i", help="Iteration to render")
    ] = None,
    quiet: Annotated[
        bool, typer.Option("--quiet", help="Suppress output")
    ] = None,
    optim_test_pose_iter: Annotated[
        int, typer.Option("--optim-test-pose-iter", help="Test pose optimization iterations")
    ] = None,
    render_path: Annotated[
        bool, typer.Option("--render-path", help="Render video path")
    ] = None,
    force_debug: Annotated[
        bool, typer.Option("--force-debug", help="Force debug mode")
    ] = None,
    unbounded: Annotated[
        bool, typer.Option("--unbounded", help="Use unbounded mesh extraction")
    ] = None,
) -> None:
    """Rendering script using YAML config with optional CLI overrides."""
    config = Path(config_path)
    if not config.exists():
        console.print(f"[red]Error: config file {config} does not exist![/red]")
        sys.exit(-1)

    with open(config, "r") as f:
        data_cf = yaml.safe_load(f)

    args = build_args_from_config(data_cf)

    # Override config values with CLI arguments if provided
    if eval is not None:
        args.eval = eval
    if skip_train is not None:
        args.skip_train = skip_train
    if skip_test is not None:
        args.skip_test = skip_test
    if skip_mesh is not None:
        args.skip_mesh = skip_mesh
    if iterations is not None:
        args.iterations = iterations
    if quiet is not None:
        args.quiet = quiet
    if optim_test_pose_iter is not None:
        args.optim_test_pose_iter = optim_test_pose_iter
    if render_path is not None:
        args.render_path = render_path
    if force_debug is not None:
        args.force_debug = force_debug
    if unbounded is not None:
        args.unbounded = unbounded

    console.print(f"Rendering {args.model_path}")

    # Create dummy parser for ParamGroup extraction
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    dataset, iteration, pipe = (
        model.extract(args),
        args.iterations,
        pipeline.extract(args),
    )

    run_rendering(args, dataset, iteration, pipe)

    console.print("\n[green]Rendering complete.[/green]")


def main_argparse():
    """Original argparse-based entry point (fallback)."""
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iterations", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_mesh", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_path", action="store_true")
    parser.add_argument(
        "--voxel_size", default=-1.0, type=float, help="Mesh: voxel size for TSDF"
    )
    parser.add_argument(
        "--depth_trunc", default=-1.0, type=float, help="Mesh: Max depth range for TSDF"
    )
    parser.add_argument(
        "--sdf_trunc", default=-1.0, type=float, help="Mesh: truncation value for TSDF"
    )
    parser.add_argument(
        "--num_cluster",
        default=50,
        type=int,
        help="Mesh: number of connected clusters to export",
    )
    parser.add_argument(
        "--unbounded",
        action="store_true",
        help="Mesh: using unbounded mode for meshing",
    )
    parser.add_argument(
        "--mesh_res",
        default=1024,
        type=int,
        help="Mesh: resolution for unbounded mesh extraction",
    )
    parser.add_argument("--optim_test_pose_iter", default=500, type=int)
    parser.add_argument("--infer_video", action="store_true")
    parser.add_argument(
        "--force_debug",
        action="store_true",
        help="Override cfg_args and enable pipeline debug",
    )
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    dataset, iteration, pipe = (
        model.extract(args),
        args.iterations,
        pipeline.extract(args),
    )

    run_rendering(args, dataset, iteration, pipe)

    print("\nRendering complete.")


if __name__ == "__main__":
    # Check if first argument looks like a config file path
    if len(sys.argv) > 1 and sys.argv[1].endswith((".cfg", ".yaml", ".yml")):
        typer.run(main_typer)
    else:
        main_argparse()
