?	X9??v@X9??v@!X9??v@	?yOY?@?yOY?@!?yOY?@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$X9??v@q???h??A+????@Y%u???*	hffff&U@2F
Iterator::Model??W?2ġ?!hu?B?D@)???Mb??1r?q?;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?]K?=??!#??-,q?@)A??ǘ???1Y???=:@:Preprocessing2U
Iterator::Model::ParallelMapV2Ǻ?????!???&@z*@)Ǻ?????1???&@z*@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?!??u???!??????0@)??ZӼ???1?n??(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipx$(~???!??m??}M@)HP?sׂ?1?@cDٿ%@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n??r?!(?????@)/n??r?1(?????@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?q????o?!)͋??p@)?q????o?1)͋??p@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?yOY?@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	q???h??q???h??!q???h??      ??!       "      ??!       *      ??!       2	+????@+????@!+????@:      ??!       B      ??!       J	%u???%u???!%u???R      ??!       Z	%u???%u???!%u???JCPU_ONLYY?yOY?@b Y      Y@q+???rS@"?
both?Your program is POTENTIALLY input-bound because 4.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?77.792% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 