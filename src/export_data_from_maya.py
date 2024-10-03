import csv
import maya.cmds as cmds

# Hardcoded attribute to export
attribute_to_export = "usr_CNT_JAW.Open_jaw"

# Check if the attribute exists
if not cmds.objExists(attribute_to_export):
    cmds.warning(f"The attribute {attribute_to_export} does not exist.")
else:
    # Get the start and end frame of the animation
    start_frame = int(cmds.playbackOptions(query=True, minTime=True))
    end_frame = int(cmds.playbackOptions(query=True, maxTime=True))
    
    # Create a file dialog to save the CSV file
    file_path = cmds.fileDialog2(fileFilter="*.csv", dialogStyle=2, caption="Save Animation Data", fileMode=0)
    
    if file_path:
        file_path = file_path[0]
        # Open the file for writing
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            
            # Write the header
            writer.writerow(["Frame", "Value"])
            
            # Write the value of the attribute at each frame to the CSV file
            for frame in range(start_frame, end_frame + 1):
                cmds.currentTime(frame, edit=True)
                value = cmds.getAttr(attribute_to_export)
                writer.writerow([frame, value])
        
        print(f"Animation data for {attribute_to_export} exported successfully to {file_path}")
    else:
        cmds.warning("No file path selected.")
