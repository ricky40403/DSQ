import copy

import torch
import torch.nn as nn
from collections import OrderedDict

import inspect

class _ReplaceFunc(object):
	"""!
	This Function replace torch functions with self-define Function.
	Inorder to get the imformation of torch model layer infomration.
	"""
	def __init__(self, ori_func, replace_func, **kwargs):
		self.torch_func = ori_func
		self.replace_func = replace_func

	def __call__(self, *args, **kwargs):		
		out = self.replace_func(self.torch_func, *args, **kwargs)
		return out


class Log(object):
	"""!
	This class use as an log to replace input tensor and store all the information
	"""
	def __init__(self):
		self.graph = OrderedDict()
		self.bottoms = OrderedDict()
		self.output_shape = OrderedDict()
		self.record_tensor_op = []
		self.cur_tensor = None
		self.cur_id = None
		self.tmp_list = None
		self.log_init()

	def __len__(self):
		"""!
		Log should be one
		"""
		return 1

	def __copy__(self):
		"""!
		copy, create new one and assign clone tensor in log
		"""
		copy_paster = Log()
		copy_paster.__dict__.update(self.__dict__)
		copy_paster.cur_tensor = self.cur_tensor.clone()
		return copy_paster

	def __deepcopy__(self, memo):
		"""!
		deepcopy, create new one and assign clone tensor in log
		"""
		copy_paster = Log()
		copy_paster.__dict__.update(self.__dict__)
		copy_paster.cur_tensor = self.cur_tensor.clone()
		return copy_paster

	def reset(self):
		"""
		This function reset all attribute in log.		
		"""
		self.graph = OrderedDict()
		self.bottoms = OrderedDict()
		self.output_shape = OrderedDict()
		self.cur_tensor = None
		self.cur_id = None
		self.tmp_list = []
		self.log_init()
	
	  
	# add data input layer to log
	def log_init(self):
		"""
		Init log attribute, set Data Layer as the first layer
		"""
		layer_id = "Data"
		self.graph[layer_id] = layer_id
		self.bottoms[layer_id] = None
		self.output_shape[layer_id] = ""
		self.cur_id = layer_id
		self.tmp_list = []

  
	# for general layer (should has only one input?)
	def putLayer(self, layer):	
		"""!
		Put genreal layer's information into log
		"""	
		# force use different address id ( prevent use same defined layer more than once, eg: bottleneck in torchvision)
		# tmp_layer = copy.deepcopy(layer)
		layer_id = id(layer)
		self.tmp_list.append(layer)
		layer_id = id(self.tmp_list[-1])
		if layer_id in self.graph:
			tmp_layer = copy.deepcopy(layer)
			self.tmp_list.append(tmp_layer)
			# layer_id = id(self.tmp_list[-1])
			layer_id = id(tmp_layer)

		self.graph[layer_id] = layer
		self.bottoms[layer_id] = [self.cur_id]
		self.cur_id = layer_id
		# del layer, tmp_layer, layer_id

	def getGraph(self):
		"""!
		This function get the layers graph from log
		"""
		return self.graph
	
	def getBottoms(self):
		"""!
		This function get the layers bottoms from log
		"""
		return self.bottoms
	
	def getOutShapes(self):
		"""!
		This function get the layers output shape from log
		"""
		return self.output_shape
	
	def getTensor(self):
		"""!
		This function get the layers current tensor (output tensor)
		"""
		return self.cur_tensor
	
	def getRecordTensorOP(self):
		"""!
		This function get the list of tensor_op record (e.g. __add__)
		"""
		return self.record_tensor_op

	def setTensor(self, tensor):
		"""!
		This function set the layer's current tensor
		and also change output shape by the input tensor
		"""		
		self.cur_tensor = tensor
		if tensor is not None:
			self.output_shape[self.cur_id] = self.cur_tensor.size()
		else:
			self.output_shape[self.cur_id] = None
	
	
	# handle tensor operation(eg: tensor.view)
	def __getattr__(self, name):
		"""!
		This function handle all the tensor operation
		"""		
		if name == "__deepcopy__" or name == "__setstate__":
			return object.__getattribute__(self, name)			
		# if get data => get cur_tensor.data
		elif name == "data":
			return self.cur_tensor.data		
		
		elif hasattr(self.cur_tensor, name):			
			def wrapper(*args, **kwargs):				
				func = self.cur_tensor.__getattribute__(name)
				out_tensor = func(*args, **kwargs)

				if type(out_tensor) == int:
					return out_tensor
				elif not isinstance(out_tensor, torch.Tensor):
					out_logs = []
					for t in out_tensor:
						out_log = copy.deepcopy(self)
						out_log.setTensor(t)						
						out_logs.append(out_log)
						
					return out_logs
				else:						
					self.cur_tensor = out_tensor
					self.output_shape[self.cur_id] = out_tensor.size() 

					return self
			# print(wrapper)
			return wrapper
			
			# return self


		else:
			return object.__getattribute__(self, name)			
	
	def __getitem__(self, idx):
		self.cur_tensor = self.cur_tensor[idx]
		self.output_shape[self.cur_id] = self.cur_tensor.size() 

		return self

	
	def __add__(self, other):
		"""!
		Log addition
		"""
		#print("add")
		# merge other branch		
		if type(other) == Log:
			self.graph.update(other.graph)
			self.bottoms.update(other.bottoms)
			self.output_shape.update(other.output_shape)
			layer_name = "add_{}".format(len(self.graph))
			self.graph[layer_name] = layer_name
			self.bottoms[layer_name] = [self.cur_id, other.cur_id]
			self.output_shape[layer_name] = self.cur_tensor.size()
			self.cur_id = layer_name
			# save memory
			del other	

			_stack = inspect.stack()
			if 'self' in _stack[1][0].f_locals:
				self.record_tensor_op.append('{}_{}_{}'.format(_stack[1][0].f_locals['self'].__class__.__name__, _stack[1].lineno, len(self.bottoms[layer_name])))
		
		return self		
	

	def __iadd__(self, other):
		"""!
		Log identity addition
		"""
		#print("iadd")		
		# merge other branch	
		if type(other) == Log:	
			self.graph.update(other.graph)
			self.bottoms.update(other.bottoms)
			self.output_shape.update(other.output_shape)
			layer_name = "iadd_{}".format(len(self.graph))
			self.graph[layer_name] = layer_name
			self.bottoms[layer_name] = [self.cur_id, other.cur_id]
			self.output_shape[layer_name] = self.cur_tensor.size()
			self.cur_id = layer_name
			# save memory
			del other		
			_stack = inspect.stack()
			if 'self' in _stack[1][0].f_locals:
				self.record_tensor_op.append('{}_{}_{}'.format(_stack[1][0].f_locals['self'].__class__.__name__, _stack[1].lineno, len(self.bottoms[layer_name])))
		return self
	

	def __sub__(self, other):
		"""!
		Log substraction
		"""
		#print("sub")
		# merge other branch
		if type(other) == Log:
			self.graph.update(other.graph)
			self.bottoms.update(other.bottoms)
			self.output_shape.update(other.output_shape)
			layer_name = "sub_{}".format(len(self.graph))
			self.graph[layer_name] = layer_name
			self.bottoms[layer_name] = [self.cur_id, other.cur_id]
			self.output_shape[layer_name] = self.cur_tensor.size()
			self.cur_id = layer_name
			# save memory
			del other
			_stack = inspect.stack()
			if 'self' in _stack[1][0].f_locals:
				self.record_tensor_op.append('{}_{}_{}'.format(_stack[1][0].f_locals['self'].__class__.__name__, _stack[1].lineno, len(self.bottoms[layer_name])))
		return self
	

	def __isub__(self, other):
		"""!
		Log identity substraction
		"""
		#print("isub")
		# merge other branch
		if type(other) == Log:
			self.graph.update(other.graph)
			self.bottoms.update(other.bottoms)
			self.output_shape.update(other.output_shape)
			layer_name = "sub_{}".format(len(self.graph))
			self.graph[layer_name] = layer_name
			self.bottoms[layer_name] = [self.cur_id, other.cur_id]
			self.output_shape[layer_name] = self.cur_tensor.size()
			self.cur_id = layer_name
			# save memory
			del other
			_stack = inspect.stack()
			if 'self' in _stack[1][0].f_locals:
				self.record_tensor_op.append('{}_{}_{}'.format(_stack[1][0].f_locals['self'].__class__.__name__, _stack[1].lineno, len(self.bottoms[layer_name])))
		return self
	

	def __mul__(self, other):
		"""!
		Log multiplication
		"""
		#print("mul")
		# merge other branch
		if type(other) == Log:
			self.graph.update(other.graph)
			self.bottoms.update(other.bottoms)
			self.output_shape.update(other.output_shape)
			layer_name = "mul_{}".format(len(self.graph))
			self.graph[layer_name] = layer_name
			self.bottoms[layer_name] = [self.cur_id, other.cur_id]
			self.output_shape[layer_name] = self.cur_tensor.size()
			self.cur_id = layer_name
			# save memory
			del other
			_stack = inspect.stack()
			if 'self' in _stack[1][0].f_locals:
				self.record_tensor_op.append('{}_{}_{}'.format(_stack[1][0].f_locals['self'].__class__.__name__, _stack[1].lineno, len(self.bottoms[layer_name])))
		return self
	

	def __imul__(self, other):
		"""!
		Log identity multiplication
		"""
		#print("imul")
		# merge other branch
		if type(other) == Log:
			self.graph.update(other.graph)
			self.bottoms.update(other.bottoms)
			self.output_shape.update(other.output_shape)
			layer_name = "mul_{}".format(len(self.graph))
			self.graph[layer_name] = layer_name
			self.bottoms[layer_name] = [self.cur_id, other.cur_id]
			self.output_shape[layer_name] = self.cur_tensor.size()
			self.cur_id = layer_name
			# save memory
			del other
			_stack = inspect.stack()
			if 'self' in _stack[1][0].f_locals:
				self.record_tensor_op.append('{}_{}_{}'.format(_stack[1][0].f_locals['self'].__class__.__name__, _stack[1].lineno, len(self.bottoms[layer_name])))
		return self

	def __div__(self, other):
		"""!
		Log division
		"""
		#print("div")
		# merge other branch
		if type(other) == Log:
			self.graph.update(other.graph)
			self.bottoms.update(other.bottoms)
			self.output_shape.update(other.output_shape)
			layer_name = "div_{}".format(len(self.graph))
			self.graph[layer_name] = layer_name
			self.bottoms[layer_name] = [self.cur_id, other.cur_id]
			self.output_shape[layer_name] = self.cur_tensor.size()
			self.cur_id = layer_name
			# save memory
			del other
			_stack = inspect.stack()
			if 'self' in _stack[1][0].f_locals:
				self.record_tensor_op.append('{}_{}_{}'.format(_stack[1][0].f_locals['self'].__class__.__name__, _stack[1].lineno, len(self.bottoms[layer_name])))
		return self

	def __idiv__(self, other):
		"""!
		Log identity division
		"""
		#print("idiv")
		# merge other branch
		if type(other) == Log:
			self.graph.update(other.graph)
			self.bottoms.update(other.bottoms)
			self.output_shape.update(other.output_shape)
			layer_name = "idiv_{}".format(len(self.graph))
			self.graph[layer_name] = layer_name
			self.bottoms[layer_name] = [self.cur_id, other.cur_id]
			self.output_shape[layer_name] = self.cur_tensor.size()
			self.cur_id = layer_name
			# save memory
			del other
			_stack = inspect.stack()
			if 'self' in _stack[1][0].f_locals:
				self.record_tensor_op.append('{}_{}_{}'.format(_stack[1][0].f_locals['self'].__class__.__name__, _stack[1].lineno, len(self.bottoms[layer_name])))
		return self

	def __truediv__(self, other):
		"""!
		Log division
		"""
		#print("truediv")
		# merge other branch
		if type(other) == Log:
			self.graph.update(other.graph)
			self.bottoms.update(other.bottoms)
			self.output_shape.update(other.output_shape)
			layer_name = "truediv_{}".format(len(self.graph))
			self.graph[layer_name] = layer_name
			self.bottoms[layer_name] = [self.cur_id, other.cur_id]
			self.output_shape[layer_name] = self.cur_tensor.size()
			self.cur_id = layer_name
			# save memory
			del other
			_stack = inspect.stack()
			if 'self' in _stack[1][0].f_locals:
				self.record_tensor_op.append('{}_{}_{}'.format(_stack[1][0].f_locals['self'].__class__.__name__, _stack[1].lineno, len(self.bottoms[layer_name])))
		
		return self

	def __itruediv__(self, other):
		"""!
		Log division
		"""
		#print("itruediv")
		# merge other branch
		if type(other) == Log:
			self.graph.update(other.graph)
			self.bottoms.update(other.bottoms)
			self.output_shape.update(other.output_shape)
			layer_name = "itruediv_{}".format(len(self.graph))
			self.graph[layer_name] = layer_name
			self.bottoms[layer_name] = [self.cur_id, other.cur_id]
			self.output_shape[layer_name] = self.cur_tensor.size()
			self.cur_id = layer_name
			# save memory
			del other
			_stack = inspect.stack()
			if 'self' in _stack[1][0].f_locals:
				self.record_tensor_op.append('{}_{}_{}'.format(_stack[1][0].f_locals['self'].__class__.__name__, _stack[1].lineno, len(self.bottoms[layer_name])))
		
		return self

	def size(self, dim=None):
		"""!
		This function return the size of the tensor by given dim

		@param dim: defult None, return as tensor.size(dim)

		@return tensor size by dim
		"""
		return self.cur_tensor.size(dim) if dim is not None else self.cur_tensor.size()



class UnitLayer(nn.Module):
	"""!
	This class is an Unit-layer act like an identity layer
	"""
	def __init__(self, ori_layer):
		super(UnitLayer, self).__init__()
		self.origin_layer = ori_layer
		

	def setOrigin(self, ori_layer):
		self.origin_layer = ori_layer


	# general layer should has only one input?
	def forward(self, log, *args):
		# prevent overwrite log for other forward flow
		cur_log = copy.deepcopy(log)
		# print(cur_log)
		cur_log.putLayer(self.origin_layer)
		
		# print(log.cur_tensor)
		log_tensor = log.getTensor()		
		# out_tensor = self.origin_layer(log_tensor).clone().detach()		
		out_tensor = self.origin_layer(log_tensor).clone()
		cur_log.setTensor(out_tensor)

		return cur_log


def dict_merge(dct, merge_dct):
	""" Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
	updating only top-level keys, dict_merge recurses down into dicts nested
	to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
	``dct``.
	:param dct: dict onto which the merge is executed
	:param merge_dct: dct merged into dct
	:return: None
	source: https://gist.github.com/angstwad/bf22d1822c38a92ec0a9
	"""
	import collections

	for k, v in merge_dct.items():
		if (k in dct and isinstance(dct[k], dict)
				and isinstance(merge_dct[k], collections.Mapping)):
			dict_merge(dct[k], v)
		else:
			dct[k] = v