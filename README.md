# Get started
This project uses `uv` for dependency management. [Check the uv documentation](https://docs.astral.sh/uv/getting-started/installation/)

# Running
 * Create a directory `input` in the project root
 * Download the input video `Unlocking Facial Recognition_ Diverse Activities Analysis.mp4`
 * Download the output video `rosto-coberto.mp4`
 * Run
    ```shell
    uv run main.py
    ```

# Report
Experiment with different backends:

* **Opencv(default)**. Some frames were not detected, like the women laid in the bed and the women with the face covered by medical bonds.
* **Retinaface**. A more robust backend, the process was veeeeery slow. We cut the video with just the frames with the face covered women, and we achieve a better recognition.