from selfsup.utils.limit_threads import *
import argparse
import os
import torch
import numpy as np
from selfsup.downstream_tasks.feature_extraction.datasets.coco import get_coco_dataloader


class FeatureExtraction():
    def __init__(self, feature_extractor, dataloader, output_path):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.feature_extractor = feature_extractor
        self.feature_extractor.to(self.device)
        self.dataloader = dataloader
        self.output_path = output_path

    def extract_features(self):
        self.feature_extractor.eval()
        all_feats =  []
        all_lbl = []
        print("Extracting features.")
        with torch.no_grad():
            for itr, (x, lbls) in enumerate(self.dataloader):
                print(f"Extraction {itr}/{len(self.dataloader)}")
                x = x.to(self.device)
                feats = self.feature_extractor(x)
                all_feats.append(feats.view(x.shape[0], -1))
                all_lbl.extend(lbls)

        all_feats = torch.cat(all_feats, dim=0).cpu().numpy()
        all_lbl = np.array(all_lbl)
        
        print("Saving extracted features ...")
        np.save(os.path.join(self.output_path, "features.npy"), all_feats)
        np.save(os.path.join(self.output_path, "labels.npy"), all_lbl)
        print(f"Number of extracted features: {all_feats.shape[0]}")


def execute(args, model):
    os.makedirs(args["output_path"], exist_ok=True)

    feature_extractor = model.feature_extractor

    if args["dataset"] == "coco2014":
        dataloader = get_coco_dataloader(args)
    else:
        raise RuntimeError("Dataset not defined")

    feature_extraction = FeatureExtraction(feature_extractor, dataloader, args["output_path"])
    feature_extraction.extract_features()


