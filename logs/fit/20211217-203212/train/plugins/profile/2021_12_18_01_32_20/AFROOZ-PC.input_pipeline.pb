$	??? ?r???
F????q??????!Ԛ?????	!       "\
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsԚ?????a??+e??A???(\???"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?q??????u????A??_?LU?*	?????lT@2F
Iterator::Model㥛? ???!>Z?t{VF@)-C??6??1
?nϊU?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat=?U?????!? G??=@)?0?*???1????)?8@:Preprocessing2U
Iterator::Model::ParallelMapV246<?R??!?0?4خ*@)46<?R??1?0?4خ*@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??H?}??!?7??1@)?j+??݃?1@?11Ӿ'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipM??St$??!¥???K@)S?!?uq{?1?哨?f @:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceU???N@s?!?:U??@)U???N@s?1?:U??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????Mbp?!?!??v?@)????Mbp?1?!??v?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 42.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	ı.n???????׊m?a??+e??!u????	!       "	!       *	!       2$	i?q??????S2??Ʃ???_?LU?!???(\???:	!       B	!       J	!       R	!       Z	!       JCPU_ONLYb 