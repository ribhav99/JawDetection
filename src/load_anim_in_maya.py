# Just like the export data file, this time use the .pt file and load the animation in Maya
import maya.cmds as cmds
import os
import torch

def load_animation_from_pt(pt_file_path):
    # Load the .pt file
    data = torch.load(pt_file_path)
    attribute_to_modify = "usr_CNT_JAW.Open_jaw"
    for i in range(len(data)):
        cmds.currentTime(i, edit=True)
        cmds.setKeyframe(attribute_to_modify, value=data[i])

load_animation_from_pt(os.path.join(os.path.dirname(__file__), "../new_outputs/KYLE-D1-012.pt"))
    
