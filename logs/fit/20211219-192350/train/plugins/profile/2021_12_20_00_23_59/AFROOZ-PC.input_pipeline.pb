$	?/?'??s2?!(??/?$???!	?c?Z??	!       "\
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails	?c?Z??J+???A??0?*??"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails/?$???HP?sׂ?A??_?LU?*	     @T@2F
Iterator::ModeljM????!?n???G@)9??v????1uk~X?@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??@??ǘ?!\??"e?=@)??ׁsF??1?q?q8@:Preprocessing2U
Iterator::Model::ParallelMapV2?HP???!?<ݚ.@)?HP???1?<ݚ.@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????????!~X?<?.@)/n????1??Hx?%@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip'???????!a?2?tkJ@)?HP?x?1?<ݚ@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n??r?!??Hx?@)/n??r?1??Hx?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice???_vOn?!O???E@)???_vOn?1O???E@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 41.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	7?[ A??s2?!(??HP?sׂ?!J+???	!       "	!       *	!       2$	?E???Ԙ?O??e?????_?LU?!??0?*??:	!       B	!       J	!       R	!       Z	!       JCPU_ONLYb 