
### Data
You can download the data json from [data](https://nuage.isir.upmc.fr/index.php/s/ACRfZgaZTp9boZ8).
These data contains annotation coming from several public datasets and their use is bounded to their corresponding licenses.

To download the images, videos and audios, please go the website of each dataset.


Our `data/` directory is organized as follows:
```
data/
    vqa/
        karpathy_train.json
        karpathy_val.json
        karpathy_test.json

        val_standard.json
        train_standard.json

    COCO/
        dataset_coco.json
        
        images/
            train2014
            val2014
    GQA/
        train.json
        testdev.json
        valid.json

        images/
            n356822.jpg
            ...
    VG/
        VG_100K/

    audiocaps/
        annotation/
            audiocaps_caption_train
            audiocaps_caption_val
            audiocaps_caption_test
        audios/
            train/
                --L22BmDI6E.wav
                ...
            test/
            val/

    MSRVTT/
        annotation/
            msrvtt_caption_train7k
            msrvtt_caption_test
            msrvtt_vqa_train
            msrvtt_vqa_val
            msrvtt_vqa_test

        videos/
            all/
                video1000.mp4
                ...

    MSVD/
        annotation/
            msvd_vqa_train.json
            msvd_vqa_val.json
            msvd_vqa_test.json

        videos/
            all/
                00jrXRMlZOY_0_10.avi
                ...
```