	8/N|�#�?8/N|�#�?!8/N|�#�?	�����1@�����1@!�����1@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$8/N|�#�?�>:u��?A�Nw�x��?Y�?����?*	X9��v&P@2F
Iterator::Model	��8�d�?!�(�l�D@)�����ڐ?1��P�z9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat B\9{g�?!]��?�>@)'������?1B#nuD9@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap������?!4JSV(5@),F]k�S�?1lR��0@:Preprocessing2U
Iterator::Model::ParallelMapV2㊋�r�?!lɔF5�/@)㊋�r�?1lɔF5�/@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��N@a�?!'���KM@)��L�nq?1�_[�?Z@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�\�	�m?!i��&+O@)�\�	�m?1i��&+O@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice0��!�j?!�ߧ�%@)0��!�j?1�ߧ�%@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 7.0% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t21.1 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9�����1@>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�>:u��?�>:u��?!�>:u��?      ��!       "      ��!       *      ��!       2	�Nw�x��?�Nw�x��?!�Nw�x��?:      ��!       B      ��!       J	�?����?�?����?!�?����?R      ��!       Z	�?����?�?����?!�?����?JCPU_ONLYY�����1@b 