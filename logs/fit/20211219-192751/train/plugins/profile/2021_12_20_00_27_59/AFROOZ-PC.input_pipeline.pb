	&䃞?*@&䃞?*@!&䃞?*@	kBGK??kBGK??!kBGK??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$&䃞?*@?D?????A??St$?@YJ+???*	43333sP@2F
Iterator::Model???_vO??!(e?~F@)j?t???1?t?^V@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??_?L??!#;?f??@)?St$????1Z%P?[:9@:Preprocessing2U
Iterator::Model::ParallelMapV2??ǘ????!???M??(@)??ǘ????1???M??(@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?0?*??!W	T??N/@)???_vO~?1(e?~&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?:pΈ??!ٚq???K@)??ZӼ?t?1?ד2? @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?J?4q?!&W?+?@)?J?4q?1&W?+?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?????g?!]H???@)?????g?1]H???@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 5.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9kBGK??>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?D??????D?????!?D?????      ??!       "      ??!       *      ??!       2	??St$?@??St$?@!??St$?@:      ??!       B      ??!       J	J+???J+???!J+???R      ??!       Z	J+???J+???!J+???JCPU_ONLYYkBGK??b 