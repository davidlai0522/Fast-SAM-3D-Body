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

# --rotate_deg -> cv2.rotate code that turns the camera image upright (+90 = counterclockwise).
_CV2_ROTATE = {90: cv2.ROTATE_90_COUNTERCLOCKWISE, -90: cv2.ROTATE_90_CLOCKWISE}
# --rotate_deg -> 3x3 that maps a point from the upright (rotated) image's camera frame back to
# the real camera's frame (OpenCV axes: x right, y down, z forward). Rotating the image by 90
# deg is the same as rolling the camera 90 deg about its optical axis.
_UNROTATE_POINTS = {
    90: np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    -90: np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
}


def rotate_cam_int(fx, fy, cx, cy, width, height, rotate_deg):
    """Intrinsics of the width x height image after cv2.rotate by rotate_deg."""
    if rotate_deg == 90:   # (u, v) -> (v, width-1-u)
        return fy, fx, cy, width - 1 - cx
    return fy, fx, height - 1 - cy, cx  # -90: (u, v) -> (height-1-v, u)


def unrotate_bbox(bbox, width, height, rotate_deg):
    """Map an x1,y1,x2,y2 box from the rotated image back to the width x height original."""
    x1, y1, x2, y2 = bbox
    if rotate_deg == 90:
        return np.array([width - 1 - y2, x1, width - 1 - y1, x2])
    return np.array([y1, height - 1 - x2, y2, height - 1 - x1])


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
    parser.add_argument(
        '--rotate_deg', type=int, default=0, choices=[0, 90, -90],
        help='Rotate each image by this much (+ = counterclockwise) before detection, so a '
             'camera mounted on its side sees people upright; outputs are rotated back into '
             'the original camera frame.',
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
        cam_int_raw = (fx, fy, cx, cy)
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
            if args.rotate_deg:
                height, width = img_cv2.shape[:2]
                img_input = cv2.cvtColor(
                    cv2.rotate(img_cv2, _CV2_ROTATE[args.rotate_deg]), cv2.COLOR_BGR2RGB,
                )
                img_cam_int = cam_int
                if cam_int is not None:
                    fx, fy, cx, cy = rotate_cam_int(*cam_int_raw, width, height, args.rotate_deg)
                    img_cam_int = torch.tensor(
                        [[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]]
                    )
            else:
                img_input, img_cam_int = image_path, cam_int
            outputs = estimator.process_one_image(
                img_input, cam_int=img_cam_int, hand_box_source=args.hand_box_source,
            )
            if args.rotate_deg:
                unrotate = _UNROTATE_POINTS[args.rotate_deg]
                for o in outputs:
                    o['pred_vertices'] = o['pred_vertices'] @ unrotate.T
                    o['pred_cam_t'] = o['pred_cam_t'] @ unrotate.T
                    o['bbox'] = unrotate_bbox(o['bbox'], width, height, args.rotate_deg)
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
