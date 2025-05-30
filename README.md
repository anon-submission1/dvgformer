# DVGFormer: Learning Camera Movement Control from Real-World Drone Videos



*"To record as is, not to create from scratch."*
![Video preview](assets/videos/teaser.gif)

Abstract: *Dynamic camera movement is a primary tool for guiding viewers’ attention and conveying emotion in video. Creating compelling camera movement demands both thoughtful shot planning and smooth camera operation. Existing AI videography methods can generate camera trajectories based on predefined rules, exemplar video clips, or textual prompts; however, they still rely on humans for shot planning. In this study, we introduce a new task called unguided camera movement generation, in which a model autonomously plans and executes camera trajectories according to captured images. To enable research on this task, we (i) curate 99k real‑world camera trajectories via 3D reconstruction and quality filtering, (ii) release a simulation benchmark covering 38 natural landscape scenes and 7 photorealistic city scans, and (iii) propose metrics that jointly measure directing quality and motion smoothness. We further present a strong baseline for this new task named DVGFormer, which generates diverse and complex camera trajectories based on captured images. DVGFormer outperforms other baselines and demonstrates sophisticated behaviors such as obstacle‑aware weaving, scenic reveals, and orbiting shots.*


## Model Checkpoint
Please refer to huggingface for checkpoint download [link](https://1drv.ms/f/c/dfb1b9d32643ecdc/EkdIKcd0zNBHsMnJBqZ-hzkBvKKxDWqatl3L8bBEGAMfCg?e=ge9wBW).

## Installation
1. **Create and activate a Conda environment**:
    ```sh
    conda create -n dvgformer python=3.10
    conda activate dvgformer
    conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
    conda install -c conda-forge ffmpeg
    pip install -r requirements.txt
    ```

2. **Download evaluation data**
   
    For real city 3D scans from Google Earth, please download from this [link](https://1drv.ms/f/c/dfb1b9d32643ecdc/EhrvMtW9ow5KrpfPJlAnJ9wBjaaYqNEKx98NOXGFteJ3pg?e=d99AG4).

    For synthetic natural scenes, you can either generate your own version from the official git repo [princeton-vl/infinigen](https://github.com/princeton-vl/infinigen) or directly download from this [link](https://1drv.ms/f/c/dfb1b9d32643ecdc/EgQWiB64W6dCsuOko_UoNQoB9Zj4cb-SSlqLFdVZITJT7Q?e=MBvCGx). Note that our version has very basic graphic settings and you might need to generate your own version if you need higher graphics. 

    After downloading the evaluation environments, your folder should look like this
    ```
    dvgformer/
    ├── infinigen/
    │   ├── arctic/
    │   ...
    │   └── snowy_mountain/
    ├── blosm/
    │   ├── himeji/
    │   ...
    │   └── sydney/
    ├── src/
    ├── README.md
    ...
    ```

3. **Download training data**
   
    We provide the Colmap 3D reconstruction results and the filtered camera movement sequences in our DroneMotion-99k dataset. You can download either a minimal dataset with 10 videos and 129 sequences [link](https://1drv.ms/u/c/dfb1b9d32643ecdc/ERIEM1bBgvVHtqgyN4T-7qoBmiHYaHcAdUUz5McREVuI_w?e=qwOBge) or the full dataset with 13,653 videos and 99,003 camera trajectories [link](https://1drv.ms/u/c/dfb1b9d32643ecdc/EcHhl1KtZrdHn4wkDJ9Kcg4BtwQCP3f3hKUHS7PArhprnw?e=SRkFjl). 

    After downloading the training data, your folder should look like this
    ```
    dvgformer/
    ├── youtube_drone_videos/
    │   ├── dataset_full.h5
    │   └── dataset_mini.h5
    ├── src/
    ├── README.md
    ...
    ```

    Due to the YouTube policy, we cannot share the video MP4s or the frames. As an alternative, we include a python script `download_videos.py` that can help you automatically download the videos and extract the frames. 
    ```sh
    python download_videos.py --hdf5_fpath youtube_drone_videos/dataset_mini.h5
    python download_videos.py --hdf5_fpath youtube_drone_videos/dataset_full.h5
    ```
    This should update your downloaded HDF5 dataset file with the video frames. 

    You can also adjust the number of workers for the download process or the frame extraction process in `download_videos.py` by specifying `--num_download_workers` or `--num_extract_workers`.


## Running DVGFormer Model
1. **Inference**:
    You can download the model checkpoint from [link](https://1drv.ms/f/c/dfb1b9d32643ecdc/EkdIKcd0zNBHsMnJBqZ-hzkBvKKxDWqatl3L8bBEGAMfCg?e=ge9wBW) and put it in the `checkpoint` folder. You can also directly load the pretrained model from this code
    ```python
    import torch
    from src.models import DVGFormerModel

    model = DVGFormerModel.from_pretrained(
        'checkpoint'
        ).to('cuda').to(torch.bfloat16)
    ```

    For blender evaluation, you can run the following script.
    ```sh
    python blender_eval.py 
    ```

2. **Train your own model**:
    We use two RTX 3090 in our experiments. Please run the following script for training your own model.
    ```sh
    bash run_gpu01.sh
    ```



