$	S'???????(LlC????HP???!Qk?w????	!       "\
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsQk?w????{?G?z??A???N@??"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?HP???HP?s??A?~j?t?X?*	??????b@2F
Iterator::Modelڬ?\mŮ?!???haD@)}гY????1ȳz?XlA@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??MbX??!?l????@@)?+e?X??1?}?V?p>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???Q???!a???4@)?{??Pk??1n?a*91@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip؁sF????!d???M@)??ZӼ???1HA%<@:Preprocessing2U
Iterator::Model::ParallelMapV2vq?-??!?v?[D@)vq?-??1?v?[D@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?J?4q?!?w?Zn@)?J?4q?1?w?Zn@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?q????o?!??"???@)?q????o?1??"???@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 32.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	*??Dؐ?????b???{?G?z??!HP?s??	!       "	!       *	!       2$	?R?!?u???f2!ܚ???~j?t?X?!???N@??:	!       B	!       J	!       R	!       Z	!       JCPU_ONLYb 