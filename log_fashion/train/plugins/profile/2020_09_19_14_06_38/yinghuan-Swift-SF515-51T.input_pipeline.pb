	qX�Q�?qX�Q�?!qX�Q�?	=����7@=����7@!=����7@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$qX�Q�?JF�v�?A�M���?Y�a����?*	������F@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatq�GR�Ð?!;��,��A@)�$�z��?1����bz=@:Preprocessing2F
Iterator::Modelޯ|�y�?!������D@)���2�?1g1��t�8@:Preprocessing2U
Iterator::Model::ParallelMapV2:�����?!�#����0@):�����?1�#����0@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�'�bd�|?!1��t�.@)���Qq?1Hp�}D"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipʧǶ8�?!UUUUU%M@)��^
j?1u�YL�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorG�ŧ h?!;��,��@)G�ŧ h?1;��,��@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlicevS�k%tg?!�YLg1@)vS�k%tg?1�YLg1@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 5.6% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t29.1 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9=����7@>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	JF�v�?JF�v�?!JF�v�?      ��!       "      ��!       *      ��!       2	�M���?�M���?!�M���?:      ��!       B      ��!       J	�a����?�a����?!�a����?R      ��!       Z	�a����?�a����?!�a����?JCPU_ONLYY=����7@b 