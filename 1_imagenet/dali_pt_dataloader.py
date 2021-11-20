from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.tfrecord as tfrec

# ImageNet1k statistics
IMAGENET_STATS = dict(
    mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
    std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
)

def dali_dataloader(
        tfrec_filenames,
        tfrec_idx_filenames,
        shard_id=0,
        num_shards=1,
        batch_size=128,
        num_threads=4,
        resize=256,
        crop=224,
        prefetch=20,
        training=True,
        gpu_aug=False,
        gpu_out=False,
        device_id=0):
    pipe = Pipeline(batch_size=batch_size,
                    num_threads=num_threads,
                    device_id=device_id if gpu_aug or gpu_out else None)
    with pipe:
        # read tfrecords, return dict(feature=TensorList/DataNode)
        inputs = fn.readers.tfrecord(
            name='Reader',
            path=tfrec_filenames,
            index_path=tfrec_idx_filenames,
            features={
                'image/encoded': tfrec.FixedLenFeature((), tfrec.string, ""),
                'image/class/label': tfrec.FixedLenFeature([1], tfrec.int64,  -1),
            },
            # sharding
            shard_id=shard_id,
            num_shards=num_shards,
            # shuffling
            random_shuffle=training,
            initial_fill=10000,
            # prefetch
            read_ahead=True,
            prefetch_queue_depth=prefetch)

        # get images
        jpegs = inputs["image/encoded"]
        decode_device = "mixed" if gpu_aug else "cpu"
        resize_device = "gpu" if gpu_aug else "cpu"

        if training:
            # decode jpeg and random crop
            images = fn.decoders.image_random_crop(jpegs,
                use_fast_idct=True,
                device=decode_device,
                output_type=types.RGB,
                random_aspect_ratio=[crop/resize, resize/crop],
                random_area=[crop/resize, 1.0],
                num_attempts=100,
                # ask HW NVJPEG to allocate memory ahead for the biggest image in the data set to avoid reallocations in runtime
                preallocate_width_hint=5980 if gpu_aug else 0,
                preallocate_height_hint=6430 if gpu_aug else 0)
            images = fn.resize(images,
                               device=resize_device,
                               resize_x=resize,
                               resize_y=resize,
                               dtype=types.FLOAT,
                               interp_type=types.INTERP_TRIANGULAR)

            # additional training transforms
            images = fn.rotate(images,
                               angle=fn.random.uniform(range=(-30, 30)),
                               keep_size=True,
                               fill_value=0)
            images = fn.noise.gaussian(images, stddev=20)
            # ... https://docs.nvidia.com/deeplearning/dali/user-guide/docs/supported_ops.html

            flip_lr = fn.random.coin_flip(probability=0.5)
        else:
            # decode jpeg and resize; caching supported
            images = fn.decoders.image(jpegs,
                                       device=decode_device,
                                       use_fast_idct=True,
                                       output_type=types.RGB,
                                      )
            images = fn.resize(images,
                               device=resize_device,
                               resize_shorter=crop,
                               dtype=types.FLOAT,
                               interp_type=types.INTERP_TRIANGULAR)
            flip_lr = False

        # center crop and normalise
        images = fn.crop_mirror_normalize(images,
                                          crop=(crop, crop),
                                          mean=IMAGENET_STATS['mean'],
                                          std=IMAGENET_STATS['std'],
                                          output_layout="CHW",
                                          mirror=flip_lr)
        label = inputs["image/class/label"] - 1  # 0-999
        if gpu_out or gpu_aug:  # transfer data to gpu
            pipe.set_outputs(images.gpu(), label.gpu())
        else:
            pipe.set_outputs(images, label)

    pipe.build()
    loader = DALIClassificationIterator(
        pipe,
        reader_name="Reader",
        last_batch_padded=False,
        auto_reset=True,
        last_batch_policy=LastBatchPolicy.DROP if training else LastBatchPolicy.PARTIAL,
    )
    return loader
