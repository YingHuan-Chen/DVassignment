	#K�X��?#K�X��?!#K�X��?	aZ~���@aZ~���@!aZ~���@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$#K�X��?D�����?A	��z���?Y�7�W���?*	G�z��L@2F
Iterator::Modelb��c?�?!82�b��F@)N{JΉ=�?1�+U�A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�P�y�?!�ێj#?@)�����?1F0��S9@:Preprocessing2U
Iterator::Model::ParallelMapV2N�M�g|?!t��~�'@)N�M�g|?1t��~�'@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��Iط�?!�s�)^�0@)�P�,y?1��+��7%@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipl#�	�?!��
�	K@)����a�m?1����A@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice"P��H�l?!�
�m
@)"P��H�l?1�
�m
@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�W\�k?!v�y|*?@)�W\�k?1v�y|*?@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 64.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9`Z~���@>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	D�����?D�����?!D�����?      ��!       "      ��!       *      ��!       2		��z���?	��z���?!	��z���?:      ��!       B      ��!       J	�7�W���?�7�W���?!�7�W���?R      ��!       Z	�7�W���?�7�W���?!�7�W���?JCPU_ONLYY`Z~���@b 