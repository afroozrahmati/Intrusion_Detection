?	??(?@??(?@!??(?@	?r?z??	@?r?z??	@!?r?z??	@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??(?@?5?;N???Aa2U0*)@Y?(\?????*	?????`@2F
Iterator::Modelz6?>W[??!?h?`?QJ@)jM????1ET?싨=@:Preprocessing2U
Iterator::Model::ParallelMapV2???_vO??!?}?:?6@)???_vO??1?}?:?6@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat????o??!?*???7@)?]K?=??1zӛ???4@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapJ+???!?B??3@)'???????1?ξ?j?0@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?Zd;??!???G@)?????w?1??.@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor	?^)?p?!??J1Aw	@)	?^)?p?1??J1Aw	@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlicea??+ei?!???J1A@)a??+ei?1???J1A@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?r?z??	@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?5?;N????5?;N???!?5?;N???      ??!       "      ??!       *      ??!       2	a2U0*)@a2U0*)@!a2U0*)@:      ??!       B      ??!       J	?(\??????(\?????!?(\?????R      ??!       Z	?(\??????(\?????!?(\?????JCPU_ONLYY?r?z??	@b Y      Y@q?*?ċ?T@"?
both?Your program is POTENTIALLY input-bound because 4.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?83.946% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 