	gDio?@gDio?@!gDio?@	a檠j8
@a檠j8
@!a檠j8
@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$gDio?@"??u????A?A?f??@Y0*??D??*	??????g@2F
Iterator::Model1?*?Թ?!?????J@)HP?s??1?o?9?1H@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?:pΈҮ?!???&??@)!?rh????1y??%??=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??A?f??!????&@)?? ?rh??1?FQ??!@:Preprocessing2U
Iterator::Model::ParallelMapV2Έ?????!t???@)Έ?????1t???@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?e??a???!%P?cYG@)?<,Ԛ?}?1LI:}??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?q????o?!?ݒ?U{ @)?q????o?1?ݒ?U{ @:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice???_vOn?!????E??)???_vOn?1????E??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 3.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9a檠j8
@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	"??u????"??u????!"??u????      ??!       "      ??!       *      ??!       2	?A?f??@?A?f??@!?A?f??@:      ??!       B      ??!       J	0*??D??0*??D??!0*??D??R      ??!       Z	0*??D??0*??D??!0*??D??JCPU_ONLYYa檠j8
@b 