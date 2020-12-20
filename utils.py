import cv2
import torch
import numpy as np


def batch_optical_flow(batch_images, batch_lens):
    gt_flows = list()
    for i in range(batch_images.shape[0] - 1):
        frame1 = cv2.cvtColor(batch_images[i].detach().cpu().numpy()[:, :, :3], cv2.COLOR_RGB2GRAY)
        frame2 = cv2.cvtColor(batch_images[i+1].detach().cpu().numpy()[:, :, :3], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        gt_flows.append(torch.FloatTensor(flow).permute(2, 0, 1).unsqueeze(0))

    gt_flows.append(torch.zeros_like(gt_flows[0]))
    gt_flows = torch.cat(gt_flows, dim=0).cuda()

    flow_weights = torch.ones((batch_images.shape[0],)).cuda()
    for i, ep_len in enumerate(batch_lens):
        idx = int(np.sum(batch_lens[:i+1]))
        flow_weights[idx-1] = 0

    return gt_flows, flow_weights

def optical_flow_loss(gt_flow, pred_flow, flow_wt):
    sq_err = (gt_flow - pred_flow)**2
    sq_err = torch.flatten(sq_err, start_dim=1)
    weighted = sq_err.mean(dim=1) * flow_wt
    return weighted.mean()
    