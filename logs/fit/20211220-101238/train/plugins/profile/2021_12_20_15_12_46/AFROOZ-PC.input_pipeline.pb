$	0*??D??O???ޠ?=?U?????!a??+e??	!       "\
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsa??+e????~j?t??A?	h"lx??"^
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails=?U???????&???A????MbP?*	     @S@2F
Iterator::Model???~?:??!qV~B??D@)A??ǘ???1S{??<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatS?!?uq??!C???gA@)??_vO??115?wL<@:Preprocessing2U
Iterator::Model::ParallelMapV2??~j?t??!?????(@)??~j?t??1?????(@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?HP???!???15?/@)?? ?rh??1?O???&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipo?ŏ1??!????cjM@) ?o_?y?1]t?E] @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??_?Lu?!L?S@)??_?Lu?1L?S@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice???_vOn?!????8@)???_vOn?1????8@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 44.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	<?R?!????˘???g???~j?t??!??&???	!       "	!       *	!       2$	%???~???????Z??????MbP?!?	h"lx??:	!       B	!       J	!       R	!       Z	!       JCPU_ONLYb 