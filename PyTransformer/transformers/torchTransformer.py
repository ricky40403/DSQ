import time
import copy
import types
import inspect 
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import pydot
from graphviz import Digraph

from .utils import _ReplaceFunc, Log, UnitLayer, dict_merge
from .quantize import QuantMeasure



class TorchTransformer(nn.Module):
	"""!
	This class handle layer swap, summary, visualization of the input model
	"""
	def __init__(self):
		super(TorchTransformer, self).__init__()
		
		self._register_dict = OrderedDict()
		self.log = Log()		
		self._raw_TrochFuncs = OrderedDict()
		self._raw_TrochFunctionals = OrderedDict()

	# register class to trans
	def register(self, origin_class, target_class):
		"""!
		This function register which class should transform to target class.		
		"""
		print("register", origin_class, target_class)

		self._register_dict[origin_class] = target_class

		pass
	
	def trans_layers(self, model, update = True):
		"""!
		This function transform layer by layers in register dictionarys

		@param model: input model to transfer

		@param update: default is True, wether to update the paramter from the orign layer or not. 
		Note that it will update matched parameters only.

		@return transfered model
		"""
		# print("trans layer")
		if len(self._register_dict) == 0:
			print("No layer to swap")
			print("Please use register( {origin_layer}, {target_layer} ) to register layer")
			return model
		else:
			for module_name in model._modules:			
				# has children
				if len(model._modules[module_name]._modules) > 0:
					self.trans_layers(model._modules[module_name], update)
				else:
					if type(getattr(model, module_name)) in self._register_dict:
						# use inspect.signature to know args and kwargs of __init__
						_sig = inspect.signature(type(getattr(model, module_name)))
						_kwargs = {}
						if update:
							for key in _sig.parameters:
								if _sig.parameters[key].default == inspect.Parameter.empty: #args 
									# assign args
									# default values should be handled more properly, unknown data type might be an issue
									if 'kernel' in key: # nn.Conv
										# _sig.parameters[key].replace(default=inspect.Parameter.empty, annotation=3)
										value = 3
									elif 'channel' in key: # nn.BatchNorm
										# _sig.parameters[key].replace(default=inspect.Parameter.empty, annotation=32)
										value = 32
									elif '_features' in key: # nn.Linear
										value = 32
									else:
										# _sig.parameters[key].replace(default=inspect.Parameter.empty, annotation=None)
										value = None
							
									_kwargs[key] = value

						_attr_dict = getattr(model, module_name).__dict__
						_layer_new = self._register_dict[type(getattr(model, module_name))](**_kwargs) # only give positional args
						dict_merge(_layer_new.__dict__, _attr_dict)

						setattr(model, module_name, _layer_new)
		return model
	
	
	

	def summary(self, model = None, input_tensor = None):
		"""!
		This function act like keras summary function
		
		@param model: input model to summary

		@param input_tensor: input data of the model to forward

		"""
		# input_tensor = torch.randn([1, 3, 224, 224])		
		# input_tensor = input_tensor.cuda()		
		

		self._build_graph(model, input_tensor)
   
		# get dicts and variables
		model_graph = self.log.getGraph()
		bottoms_graph = self.log.getBottoms()
		output_shape_graph = self.log.getOutShapes()
		# store top names for bottoms
		topNames = OrderedDict()		
		totoal_trainable_params = 0
		total_params = 0
		# loop graph
		print("##########################################################################################")
		line_title = "{:>5}| {:<15} | {:<15} {:<25} {:<15}".format("Index","Layer (type)", "Bottoms","Output Shape", "Param #")
		print(line_title)
		print("---------------------------------------------------------------------------")	

		
		for layer_index, key in enumerate(model_graph):	
			
			# data layer
			if bottoms_graph[key] is None:
				# Layer information
				layer = model_graph[key]
				layer_type = layer.__class__.__name__
				if layer_type == "str":
					layer_type = key
				else:
					layer_type = layer.__class__.__name__ + "_{}".format(layer_index)
				
				topNames[key] = layer_type

				# Layer Output shape
				output_shape = "[{}]".format(tuple(output_shape_graph[key]))
				
				# Layer Params
				param_weight_num = 0				
				if hasattr(layer, "weight") and hasattr(layer.weight, "size"):
					param_weight_num += torch.prod(torch.LongTensor(list(layer.weight.size())))
					if layer.weight.requires_grad:
						totoal_trainable_params += param_weight_num
				if hasattr(layer, "bias") and hasattr(layer.weight, "bias"):
					param_weight_num += torch.prod(torch.LongTensor(list(layer.bias.size())))				
					if layer.bias.requires_grad:
						totoal_trainable_params += param_weight_num
				
				total_params += param_weight_num
				
				new_layer = "{:5}| {:<15} | {:<15} {:<25} {:<15}".format(layer_index, layer_type, "", output_shape, param_weight_num)
				print(new_layer)
				
			else:
				# Layer Information
				layer = model_graph[key]
				layer_type = layer.__class__.__name__
				
				# add, sub, mul...,etc. (custom string)
				if layer_type == "str":
					# the key should be XXX_{idx_prevent_duplicate}
					tmp_key = key.split("_")
					tmp_key[-1] = "_{}".format(layer_index)	
					tmp_key = "".join(tmp_key)
					layer_type = tmp_key
				else:
					layer_type = layer.__class__.__name__ + "_{}".format(layer_index)

				topNames[key] = layer_type

				# Layer Bottoms
				bottoms = []
				for b_key in bottoms_graph[key]:
					bottom = topNames[b_key]				
					bottoms.append(bottom)
				
				# Layer Output Shape
				if key in output_shape_graph:
					output_shape = "[{}]".format(tuple(output_shape_graph[key]))
				else:
					output_shape = "None"
				
				# Layer Params
				param_weight_num = 0				
				if hasattr(layer, "weight") and hasattr(layer.weight, "size"):
					param_weight_num += torch.prod(torch.LongTensor(list(layer.weight.size())))
					if layer.weight.requires_grad:
						totoal_trainable_params += param_weight_num
				if hasattr(layer, "bias") and hasattr(layer.weight, "bias"):
					param_weight_num += torch.prod(torch.LongTensor(list(layer.bias.size())))				
					if layer.bias.requires_grad:
						totoal_trainable_params += param_weight_num			
				total_params += param_weight_num
				
				# Print (one bottom a line)
				for idx, b in enumerate(bottoms):					
					# if more than one bottom, only print bottom
					if idx == 0:						
						new_layer = "{:>5}| {:<15} | {:<15} {:<25} {:<15}".format(layer_index, layer_type, b, output_shape, param_weight_num)				
					else:
						new_layer = "{:>5}| {:<15} | {:<15} {:<25} {:<15}".format("", "", b, "", "")
					print(new_layer)
			print("---------------------------------------------------------------------------")
		
		
		# total information
		print("==================================================================================")
		print("Total Trainable params: {} ".format(totoal_trainable_params))
		print("Total Non-Trainable params: {} ".format(total_params - totoal_trainable_params))
		print("Total params: {} ".format(total_params))
  
		# del model_graph, bottoms_graph, output_shape_graph, topNames
		# return model

	def visualize(self, model = None, input_tensor = None, save_name = None, graph_size = 30):
		"""!
		This functin visualize the model architecture

		@param model: input model to summary

		@param input_tensor: input data of the model to forward

		@param save_name: if save_name is not None, it will save as '{save_name}.png'

		@param graph_size: graph_size for graphviz, to help increase the resolution of the output graph

		@return dot, graphviz's Digraph element
		"""
		# input_tensor = torch.randn([1, 3, 224, 224])
		# model_graph = self.log.getGraph()
		
		# if graph empty		
		if model is None:
			# check if use self modules
			if len(self._modules) > 0:
				self._build_graph(self, input_tensor)	
			else:
				raise ValueError("Please input model to visualize")
		else:
			self._build_graph(model, input_tensor)
		
		# graph 
		node_attr = dict(style='filled',
						 shape='box',
						 align='left',
						 fontsize='30',
						 ranksep='0.1',
						 height='0.2')
		
		dot = Digraph(node_attr=node_attr, graph_attr=dict(size="{},{}".format(graph_size, graph_size)))	

		# get dicts and variables
		model_graph = self.log.getGraph()		
		bottoms_graph = self.log.getBottoms()
		output_shape_graph = self.log.getOutShapes()
		topNames = OrderedDict()
		
		for layer_index, key in enumerate(model_graph):
			# Input Data layer
			if bottoms_graph[key] is None:
				layer = model_graph[key]
				layer_type = layer.__class__.__name__				
				# add, sub, mul...,etc. (custom string)
				if layer_type == "str":
					layer_type = key
				else:
					layer_type = layer.__class__.__name__ + "_{}".format(layer_index)
				
				output_shape = "{}".format(tuple(output_shape_graph[key]))
				topNames[key] = layer_type
				output_shape = "[{}]".format(tuple(output_shape_graph[key]))
				layer_type = layer_type + "\nShape: " + output_shape

				dot.node(str(key), layer_type, fillcolor='orange')			
			else:
				# Layer Information
				layer = model_graph[key]
				layer_type = layer.__class__.__name__				
				# add, sub, mul...,etc. (custom string)
				if layer_type == "str":
					# the key should be XXX_{idx_prevent_duplicate}
					tmp_key = key.split("_")
					tmp_key[-1] = "_{}".format(layer_index)	
					tmp_key = "".join(tmp_key)
					layer_type = tmp_key
				else:
					layer_type = layer.__class__.__name__ + "_{}".format(layer_index)
				
				topNames[key] = layer_type
				# layer_type = layer_type
				# print("Layer: {}".format(layer_type))
				# print("Key: {}".format(key))
				# add bottoms

				layer_type = layer_type + "\nBottoms: "
				for b_key in bottoms_graph[key]:
					layer_type = layer_type + topNames[b_key] + "\n" 
				
				output_shape = "[{}]".format(tuple(output_shape_graph[key]))
				layer_type = layer_type + "Shape: " + output_shape

				dot.node(str(key), layer_type, fillcolor='orange')				
				# link bottoms
				# print("Bottoms: ")
				for bot_key in bottoms_graph[key]:
					# print(bot_key)
					dot.edge(str(bot_key), str(key))				
		
		# return graph
		if save_name is not None:
			(graph,) = pydot.graph_from_dot_data(dot.source)
			graph.write_png(save_name + ".png" )
		return dot

	def _build_graph(self, model, input_tensor = None):

		if input_tensor is None:
			raise ValueError("Please set input tensor")

		# reset log
		self.log = Log()		
		# add Data input
		self.log.setTensor(input_tensor)		


		# tmp_model = self._trans_unit(copy.deepcopy(model))
		self._trans_unit(model)
		# print(tmp_model)
		
		for f in dir(torch):

			# if private function, pass
			if f.startswith("_") or "tensor" == f:
				continue
			if isinstance(getattr(torch, f) ,types.BuiltinMethodType) or isinstance(getattr(torch, f) ,types.BuiltinFunctionType):
				self._raw_TrochFuncs[f] = getattr(torch, f)
				setattr(torch, f, _ReplaceFunc(getattr(torch,f), self._torchFunctions))
    
		for f in dir(F):
			# if private function, pass
			if f.startswith("_"):
				continue

			if isinstance(getattr(F, f) ,types.BuiltinMethodType) or isinstance(getattr(F, f) ,types.BuiltinFunctionType) or isinstance(getattr(F, f) ,types.FunctionType):				
				self._raw_TrochFunctionals[f] = getattr(F, f)
				setattr(F, f, _ReplaceFunc(getattr(F,f), self._torchFunctionals))
				

		self.log = model.forward(self.log)
		if type(self.log) == tuple: #multiple output, just pick the first (all contains the same graph)
			self.log = self.log[0]	
		# self.log = tmp_model.forward(self.log)		

		self._restore_unit(model)

		# reset back 
		for f in self._raw_TrochFuncs:
			setattr(torch, f, self._raw_TrochFuncs[f])
   
		for f in self._raw_TrochFunctionals:
			setattr(F, f, self._raw_TrochFunctionals[f])

		# del tmp_model
		
	def _trans_unit(self, model):
		# print("TRNS_UNIT")
		for module_name in model._modules:
			if type(model._modules[module_name]) == UnitLayer:
				continue
			# has children
			if len(model._modules[module_name]._modules) > 0 and\
				not (len(model._modules[module_name]._modules) == 1 and type(list(model._modules[module_name]._modules.values())[0]) == QuantMeasure):
				model._modules[module_name] = self._trans_unit(model._modules[module_name])
			else:
				unitlayer = UnitLayer(getattr(model, module_name))
				setattr(model, module_name, unitlayer)

		return model

	
	def _restore_unit(self, model):
		# print("restore_UNIT")
		for module_name in model._modules:
			# has children
			if type(model._modules[module_name]) == UnitLayer:
				# unitlayer = UnitLayer(getattr(model, module_name))
				setattr(model, module_name, getattr(getattr(model, module_name), 'origin_layer'))

			elif len(model._modules[module_name]._modules) > 0 and\
				not (len(model._modules[module_name]._modules) == 1 and type(list(model._modules[module_name]._modules.values())[0]) == QuantMeasure):
				model._modules[module_name] = self._restore_unit(model._modules[module_name])

		return model
	
	def _torchFunctions(self, raw_func, *args, **kwargs):
		"""!
		The replaced torch function (eg: torch.{function}) will go here
		"""
		# print("Torch function")
		function_name = raw_func.__name__
  
		# torch function may has no input
		# so check first
		
		if len(args) > 0:
			logs = args[0]
			cur_args = args[1:]
		elif len(kwargs) > 0:
			
			return raw_func(**kwargs)
		else:			
			return raw_func()

		# check is user used or in torch function call
		is_tensor_in = False
		# tensor input		
		# multi tensor input
		if isinstance(logs, tuple) and (type(logs[0]) == torch.Tensor):
			cur_inputs = logs
			is_tensor_in = True
			return raw_func(*args, **kwargs)
		# single tensor input
		elif (type(logs) == torch.Tensor):
			
			cur_inputs = logs	
			is_tensor_in = True	
			# print(*args)
			# print(**kwargs)
			return raw_func(*args, **kwargs)
		elif (type(logs) == nn.Parameter):
			cur_inputs = logs	
			is_tensor_in = True				
			return raw_func(*args, **kwargs)
		# log input
		else:			
			# multi inputs
			bottoms = []
			cur_inputs = []				
	
			if isinstance(logs, tuple) or isinstance(logs, list):
				# may use origin input log as others' input 
				# eg: densenet in torchvision 0.4.0
				cur_log = copy.deepcopy(logs[0])
				for log in logs:					
					cur_inputs.append(log.cur_tensor)
					# print(log.cur_tensor.size())
					bottoms.append(log.cur_id)
					# update informations
					cur_log.graph.update(log.graph)					
					cur_log.bottoms.update(log.bottoms)
					cur_log.output_shape.update(log.output_shape)
				cur_inputs = tuple(cur_inputs)
			# one input
			else:
				# print(args)
				# print(kwargs)
				cur_log = logs
				cur_inputs = cur_log.cur_tensor
				bottoms.append(cur_log.cur_id)
				
		# replace logs to tensor as function inputs to get output tensor
		args = list(args)
		args[0] = cur_inputs
		args = tuple(args)
		# send into origin functions
		#out_tensor = raw_func(*args, **kwargs).clone().detach()
		out_tensor = raw_func(*args, **kwargs).clone()
		
		# if function call, just return out tensor
		if is_tensor_in:
			return out_tensor

		# most multi input change to one output
		# most multi output has one input
		# if shape change
		# store theese types of  opreation as a layer
		if isinstance(logs, tuple) or isinstance(logs, list) or isinstance(out_tensor, tuple) or (logs.cur_tensor.size() != out_tensor.size()):
			layer_name = "torch.{}_{}".format(function_name, len(cur_log.graph))
			cur_log.graph[layer_name] = layer_name
			cur_log.bottoms[layer_name] = bottoms
			cur_log.cur_id = layer_name
			cur_log.record_tensor_op.append('torch_{}_{}_{}'.format(function_name, inspect.stack()[2].lineno, len(bottoms)))
		
		# multi output
		if not isinstance(out_tensor , torch.Tensor):
			# print("multi output")				
			out_logs = []
			for t in out_tensor:				
				out_log = copy.deepcopy(cur_log)
				out_log.setTensor(t)			
				out_logs.append(out_log)
			
			# sometimes will has (out, ) and this lens is >1
			if len(out_logs) == 1:
				out_logs = out_logs[0]
			return out_logs

		else:
			cur_log.setTensor(out_tensor)   			
			return cur_log
		
	# torch.functionals
	def _torchFunctionals(self, raw_func, *args, **kwargs):	
		"""!
		The replaced torch.functional function (eg: F.{function}) will go here
		"""
		# print("Functional")
		function_name = raw_func.__name__		
		# print(raw_func.__name__)		

		# functional has input expect affine_grid
		if function_name == "affine_grid":
			pass
		else:
			logs = args[0]
			cur_args = args[1:]
		
		# check is user used or in torch function call
		is_tensor_in = False
		# tensor input
		if (len(logs) > 1) and (type(logs[0]) == torch.Tensor):
			# print(logs[0].size(), logs[1].size())
			cur_inputs = logs
			is_tensor_in = True
			out = raw_func(*args, **kwargs)
			# print("Functional return : {}".format(out.size()))
			return raw_func(*args, **kwargs)

		elif (len(logs) ==1) and (type(logs) == torch.Tensor):					
			cur_inputs = logs	
			is_tensor_in = True			
			out = raw_func(*args, **kwargs)
			# print("Functional return : {}".format(out.size()))
			return raw_func(*args, **kwargs)
		
		# log input
		else:			
			# multi inputs
			bottoms = []
			cur_inputs = []				
			if len(logs) > 1:				
				cur_log = logs[0]
				for log in logs:					
					cur_inputs.append(log.cur_tensor)
					bottoms.append(log.cur_id)
					# update informations
					cur_log.graph.update(log.graph)					
					cur_log.bottoms.update(log.bottoms)
					cur_log.output_shape.update(log.output_shape)
				cur_inputs = tuple(cur_inputs)
			# one input
			else:
				cur_log = logs
				cur_inputs = cur_log.cur_tensor
				bottoms.append(cur_log.cur_id)

		
			
		# replace logs to tensor as function inputs to get output tensor
		args = list(args)
		args[0] = cur_inputs
		args = tuple(args)
		# send into origin functions
		#out_tensor = raw_func(*args, **kwargs).clone().detach()
		out_tensor = raw_func(*args, **kwargs).clone()
		
		# if function call, just return out tensor
		if is_tensor_in:
			return out_tensor

		# if log input and is function type, store as an layer
		if isinstance(raw_func, types.FunctionType):			
			# use multiple address as name to prevent duplicate address
			layer_name = "F.{}_{}{}{}".format(function_name, id(out_tensor), id(args), id(kwargs))			
			# replace with new address if still duplicate
			while layer_name in cur_log.graph:
			#if layer_name in cur_log.graph:
				# tmp_list = []
				# tmp_list.append(out_tensor)
				# tmp_tensor = copy.deepcopy(tmp_list[-1])
				# tmp_tensor = tmp_list[-1].clone()				
				tmp_tensor = torch.tensor([0])
				
				# should not duplicate again?
				# layer_name = layer_name.split('.')[0] + "F" + ".{}_{}{}{}".format(function_name, id(tmp_tensor), id(args), id(kwargs))				
				layer_name = "F.{}_{}{}{}{}".format(function_name, id(tmp_tensor), id(args), id(kwargs), int((time.time()*100000)%1000000))

			cur_log.graph[layer_name] = layer_name				
			cur_log.bottoms[layer_name] = bottoms
			cur_log.cur_id = layer_name			
			cur_log.record_tensor_op.append('F_{}_{}_{}'.format(function_name, inspect.stack()[2].lineno, len(bottoms)))
		
		# if multi-output
		# if len(out_tensor) > 1:
		if not isinstance(out_tensor, torch.Tensor):
			out_logs = []
			for t in out_tensor:				
				out_log = copy.deepcopy(cur_log)
				out_log.setTensor(t)			
				out_logs.append(out_log)	
			
			return out_logs
		else:			
			cur_log.setTensor(out_tensor)
			return cur_log


