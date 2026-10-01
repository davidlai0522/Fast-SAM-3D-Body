# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Persistent SAM 3D Body worker: load the model once, then fit one image per stdin line.

Protocol (line-delimited JSON):
  worker -> parent, once ready:  {"ready": true}
  parent -> worker, per request: {"image_path": "...", "output_dir": "...", "image_name": "..."}
  worker -> parent, per request: {"mesh_path": "...", "sam3d_output_path": "..."} (sam3d_output_path
                                  holds that person's pred_vertices/pred_cam_t as an .npz, for
                                  downstream MHR->SMPL-X conversion) or {"mesh_path": null} (no
                                  person) or {"error": "..."}

All of setup_sam_3d_body()/process_one_image()'s own stdout noise (prints, tqdm, CUDA/TF
logs) is redirected to stderr so it can't corrupt the stdout protocol channel; the parent
should leave this process's stderr connected to its own logs for visibility.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

# Redirect the process's real stdout (the pipe back to the parent) to a private fd, then
# alias fd 1 (stdout) to fd 2 (stderr) so every other print/log in this process -- ours or
# a library's -- goes to stderr instead. Only send() below writes to the real stdout.
_protocol_out = os.fdopen(os.dup(1), 'w', buffering=1)
os.dup2(2, 1)
sys.stdout = sys.stderr


def send(obj):
    _protocol_out.write(json.dumps(obj) + '\n')
    _protocol_out.flush()


parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, parent_dir)

import cv2  # noqa: E402

from notebook.utils import save_mesh_results, setup_sam_3d_body  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='facebook/sam-3d-body-dinov3')
    parser.add_argument('--detector_model', default='./checkpoints/yolo/yolo11n.pt')
    parser.add_argument('--local_checkpoint', default='./checkpoints/sam-3d-body-dinov3')
    parser.add_argument('--hand_box_source', default='body_decoder')
    parser.add_argument(
        '--cam_int', default='',
        help="Known camera intrinsics 'fx,fy,cx,cy'. If set, MoGe2 FOV estimator is not "
             "loaded (saves GPU memory) and these intrinsics are used for every image.",
    )
    args = parser.parse_args()

    estimator = setup_sam_3d_body(
        hf_repo_id=args.model,
        detector_name='yolo',
        detector_model=args.detector_model,
        local_checkpoint_path=args.local_checkpoint,
        fov_name='' if args.cam_int else 'moge2',
    )
    cam_int = None
    if args.cam_int:
        fx, fy, cx, cy = (float(v) for v in args.cam_int.split(','))
        cam_int = torch.tensor([[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]])
    send({'ready': True})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            image_path = request['image_path']
            output_dir = request['output_dir']
            image_name = request.get('image_name', 'frame')

            img_cv2 = cv2.imread(image_path)
            outputs = estimator.process_one_image(
                image_path, cam_int=cam_int, hand_box_source=args.hand_box_source,
            )
            if not outputs:
                send({'mesh_path': None})
                continue

            os.makedirs(output_dir, exist_ok=True)
            ply_files = save_mesh_results(
                img_cv2, outputs, estimator.faces, output_dir, image_name,
            )
            if not ply_files:
                send({'mesh_path': None})
                continue

            # save_mesh_results() exports ply_files[0] for the first person in outputs
            # with valid (non-empty, non-NaN) vertices -- find that same person here so
            # its pred_vertices/pred_cam_t can be handed to the MHR->SMPL-X converter.
            person = next(
                o for o in outputs
                if o['pred_vertices'] is not None and len(o['pred_vertices']) > 0
                and not np.any(np.isnan(o['pred_vertices']))
                and not np.any(np.isnan(o['pred_cam_t']))
            )
            sam3d_output_path = os.path.join(output_dir, f'{image_name}_sam3d_output.npz')
            np.savez(
                sam3d_output_path,
                pred_vertices=person['pred_vertices'],
                pred_cam_t=person['pred_cam_t'],
            )
            send({'mesh_path': ply_files[0], 'sam3d_output_path': sam3d_output_path})
        except Exception as exc:
            send({'error': str(exc)})


if __name__ == '__main__':
    main()
