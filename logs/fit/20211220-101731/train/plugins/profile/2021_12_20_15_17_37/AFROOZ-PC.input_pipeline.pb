$	??%䃞??????q??X9??v???!???1段?	!       "\
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails???1段????????AB`??"۩?"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsX9??v??????B?i??A??_?LU?*	fffff?S@2F
Iterator::Model????Mb??!Ac(2[D@)????????1?_~??@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatO??e?c??!?fҕ?AB@)?~j?t???1ㄔ<ˈ>@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapǺ?????!????,@)vq?-??17??T$@:Preprocessing2U
Iterator::Model::ParallelMapV2y?&1?|?!ڢV???!@)y?&1?|?1ڢV???!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip$????ۧ?!????ͤM@)S?!?uq{?1?R9h`!@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorU???N@s?!?#A?'?@)U???N@s?1?#A?'?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceF%u?k?!?℔<?@)F%u?k?1?℔<?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 55.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	??6?[??1??j????B?i??!???????	!       "	!       *	!       2$	??ݓ??????2?ϡ???_?LU?!B`??"۩?:	!       B	!       J	!       R	!       Z	!       JCPU_ONLYb 