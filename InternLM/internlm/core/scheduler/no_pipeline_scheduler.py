#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# adopted from https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/engine

from typing import Any, Callable, Iterable, List, Optional

import torch

from internlm.core.context import global_context as gpc
from internlm.core.engine import Engine, KDEngine
from internlm.utils.common import conditional_context
from internlm.utils.timeout import llm_timeout
from collections import defaultdict
from .base_scheduler import BaseScheduler, SchedulerHook

import numpy as np

class NonPipelineScheduler(BaseScheduler):
    """A helper schedule class for no pipeline parallelism running environment.
    During one process, it loads a batch of dataset and feeds it to the model.
    After getting the output and calculating the loss, it will use :meth:`step`
    to update the parameters if it is in training mode.

    Args:
        data_process_func (Callable, optional): The preprocessing function which receives a batch of data
            and returns a tuple in the form of (data, label), and it will be executed in load_batch.
        gradient_accumulation_steps(int, optional): the steps of gradient accumulation, 1 for disable
            gradient accumulation.

    Examples:
        >>> # this shows an tools of customized data_process_func
        >>> def data_process_func(dataloader_output):
        >>>     item1, item2, item3 = dataloader_output
        >>>     data = (item1, item2)
        >>>     label = item3
        >>>     return data, label
    """

    def __init__(
        self,
        data_process_func: Callable = None,
        gradient_accumulation_size: int = 1,
        scheduler_hooks: Optional[List[SchedulerHook]] = None,
    ):
        self._grad_accum_size = gradient_accumulation_size
        self._grad_accum_offset = 0

        self._hooks = scheduler_hooks

        super().__init__(data_process_func)

    def pre_processing(self, engine: Engine):
        """Performs actions before running the schedule.

        Args:
           engine (internlm.core.Engine): InternLM engine for training and inference.
        """
        pass

    def _call_hooks(self, func_name: str, *args, **kwargs) -> None:
        for hook in self._hooks:
            getattr(hook, func_name)(self, *args, **kwargs)

    def _load_accum_batch(self, data: Any, label: Any):
        """Loads a batch of data and label for gradient accumulation.

        Args:
            data (Any): The data to be loaded.
            label (Any): The label to be loaded.
        """

        _data, _label = self._load_micro_batch(
            data=data, label=label, offset=self._grad_accum_offset, micro_bsz=self._grad_accum_batch_size
        )
        self._grad_accum_offset += self._grad_accum_batch_size

        if self.data_process_func:
            _data["input_ids"] = self.data_process_func(_data["input_ids"], _data["cu_seqlens"])
            _label = self.data_process_func(_label, _data["cu_seqlens"])
            _data.pop("cu_seqlens")
            _data.pop("indexes")

        return _data, _label

    def _train_one_batch(
        self,
        data: Any,
        label: Any,
        engine: Engine,
        forward_only: bool = False,
        return_loss: bool = True,
        scale_loss: int = 1,
    ):
        """Trains one batch of data.

        Args:
            data (Any): The data to be trained.
            label (Any): The label for the data.
            engine (internlm.core.Engine): InternLM engine for training and inference.
            forward_only (bool, optional): If True, the model is run for the forward pass, else back propagation will
                be executed.
            return_loss (bool, optional): Loss will be returned if True.
            scale_loss (int, optional): The scale factor for the loss.
        """

        # forward
        with conditional_context(torch.no_grad(), enable=forward_only):
            self._call_hooks("before_forward", data)
            output = self._call_engine(engine, data)
            self._call_hooks("after_forward", output)

            self._call_hooks("post_helper_func", output, label)

            if return_loss:
                self._call_hooks("before_criterion", output, label)
                loss = self._call_engine_criterion(engine.criterion, output, label)
                self._call_hooks("after_criterion", loss)
                loss /= scale_loss

        # backward
        if not forward_only:
            self._call_hooks("before_backward", None, None)
            engine.backward(loss)
            self._call_hooks("after_backward", None)

        if not return_loss:
            loss = None

        return output, dict(loss=loss)

    @llm_timeout(func_name="nopp_forward_backward_step")
    def forward_backward_step(
        self,
        engine: Engine,
        data_iter: Iterable,
        forward_only: bool = False,
        return_loss: bool = True,
        return_output_label: bool = True,
        global_iteration: int = 1,
    ):
        """The process function that loads a batch of dataset and feeds it to the model.
        The returned labels and loss will None if :attr:`return_loss` is False.

        Args:
            engine (internlm.core.Engine): InternLM engine for training and inference.
            data_iter (Iterable): Dataloader as the form of an iterator, obtained by calling iter(dataloader).
            forward_only (bool, optional):
                If True, the model is run for the forward pass, else back propagation will be executed.
            return_loss (bool, optional): Loss will be returned if True.
            return_output_label (bool, optional): Output and label will be returned if True.

        Returns:
            Tuple[:class:`torch.Tensor`]: A tuple of (output, label, loss), loss and label could be None.
        """
        assert (
            forward_only or return_loss
        ), "The argument 'return_loss' has to be True when 'forward_only' is False, but got False."

        batch_data, batch_size = engine.load_batch(data_iter)

        assert (
            batch_size % self._grad_accum_size == 0
        ), f"batch_size:{batch_size} must be an integer multiple of gradient accumulation steps:{self._grad_accum_size}"
        self._grad_accum_batch_size = batch_size // self._grad_accum_size

        data, label = batch_data

        loss = defaultdict(int) if return_loss else None
        outputs = []
        labels = []

        # reset accumulation microbatch offset
        self._grad_accum_offset = 0

        for _current_accum_step in range(self._grad_accum_size):
            if _current_accum_step == self._grad_accum_size - 1:
                engine.optimizer.skip_grad_reduce = False
            else:
                engine.optimizer.skip_grad_reduce = True

            _data, _label = self._load_accum_batch(data, label)

            _output, _loss = self._train_one_batch(
                _data, _label, engine, forward_only, return_loss, self._grad_accum_size
            )

            if return_loss:
                for k in _loss:
                    loss[k] += _loss[k]
            if return_output_label:
                outputs.append(_output)
                labels.append(_label)

        if not return_output_label:
            outputs, labels = None, None

        return outputs, labels, loss


class NonPipelineScheduler_motion_regression(BaseScheduler):
    """A helper schedule class for no pipeline parallelism running environment.
    During one process, it loads a batch of dataset and feeds it to the model.
    After getting the output and calculating the loss, it will use :meth:`step`
    to update the parameters if it is in training mode.

    Args:
        data_process_func (Callable, optional): The preprocessing function which receives a batch of data
            and returns a tuple in the form of (data, label), and it will be executed in load_batch.
        gradient_accumulation_steps(int, optional): the steps of gradient accumulation, 1 for disable
            gradient accumulation.

    Examples:
        >>> # this shows an tools of customized data_process_func
        >>> def data_process_func(dataloader_output):
        >>>     item1, item2, item3 = dataloader_output
        >>>     data = (item1, item2)
        >>>     label = item3
        >>>     return data, label
    """

    def __init__(
        self,
        data_process_func: Callable = None,
        gradient_accumulation_size: int = 1,
        scheduler_hooks: Optional[List[SchedulerHook]] = None,
        momentum_scheduler: Optional[np.ndarray] = None,
    ):
        self._grad_accum_size = gradient_accumulation_size
        self._grad_accum_offset = 0

        self._hooks = scheduler_hooks

        super().__init__(data_process_func)

        self.momentum_scheduler = momentum_scheduler

    def pre_processing(self, engine: Engine):
        """Performs actions before running the schedule.

        Args:
           engine (internlm.core.Engine): InternLM engine for training and inference.
        """
        pass

    def _call_hooks(self, func_name: str, *args, **kwargs) -> None:
        for hook in self._hooks:
            getattr(hook, func_name)(self, *args, **kwargs)

    def _load_accum_batch(self, data: Any, label: Any):
        """Loads a batch of data and label for gradient accumulation.

        Args:
            data (Any): The data to be loaded.
            label (Any): The label to be loaded.
        """

        _data, _label = self._load_micro_batch(
            data=data, label=label, offset=self._grad_accum_offset, micro_bsz=self._grad_accum_batch_size
        )
        self._grad_accum_offset += self._grad_accum_batch_size

        if self.data_process_func:
            _data["input_ids"] = self.data_process_func(_data["input_ids"], _data["cu_seqlens"])
            _label = self.data_process_func(_label, _data["cu_seqlens"])
            _data.pop("cu_seqlens")
            _data.pop("indexes")

        return _data, _label

    def _train_one_batch(
        self,
        data: Any,
        label: Any,
        engine: Engine,
        forward_only: bool = False,
        return_loss: bool = True,
        scale_loss: int = 1,
        global_iteration: int = 1,
    ):
        """Trains one batch of data.

        Args:
            data (Any): The data to be trained.
            label (Any): The label for the data.
            engine (internlm.core.Engine): InternLM engine for training and inference.
            forward_only (bool, optional): If True, the model is run for the forward pass, else back propagation will
                be executed.
            return_loss (bool, optional): Loss will be returned if True.
            scale_loss (int, optional): The scale factor for the loss.
        """

        # forward
        with conditional_context(torch.no_grad(), enable=forward_only):
            self._call_hooks("before_forward", data)
            output = self._call_engine(engine, data)
            # import pdb; pdb.set_trace()
            self._call_hooks("after_forward", output)

            self._call_hooks("post_helper_func", output, label)


            self._call_hooks("before_criterion", output, label)
            loss_latent = self._call_engine_criterion(engine.criterion, output, label)

            loss_pose, loss_conf = self._call_engine_criterion(engine.motion_criterion, output, data['poses'])

            # copy the loss for logging
            loss_pose_raw = loss_pose.detach()
            loss_conf_raw = loss_conf.detach()
            loss_latent_raw = loss_latent.detach()

            loss_latent = gpc.config.motion_config['latent_weight'] * loss_latent

            loss_pose = gpc.config.motion_config['reg_weight'] * loss_pose

            loss_gt = loss_pose + loss_conf
            loss_gt = gpc.config.motion_config['gt_weight'] * loss_gt * self.momentum_scheduler[global_iteration]

            self._call_hooks("after_criterion", loss_latent)
            loss_latent /= scale_loss
            loss_pose /= scale_loss
            loss_conf /= scale_loss

            loss_pose_raw /= scale_loss
            loss_conf_raw /= scale_loss
            loss_latent_raw /= scale_loss

            loss = loss_gt + loss_latent

        # backward
        if not forward_only:
            self._call_hooks("before_backward", None, None)
            engine.backward(loss)
            self._call_hooks("after_backward", None)


        return output, dict(loss=loss, loss_latent_raw=loss_latent_raw, loss_pose_raw=loss_pose_raw, loss_conf_raw=loss_conf_raw,
                            loss_latent=loss_latent, loss_pose=loss_pose, loss_conf=loss_conf,loss_gt=loss_gt)

    @llm_timeout(func_name="nopp_forward_backward_step")
    def forward_backward_step(
        self,
        engine: Engine,
        data_iter: Iterable,
        forward_only: bool = False,
        return_loss: bool = True,
        return_output_label: bool = True,
        global_iteration: int = 1,
    ):
        """The process function that loads a batch of dataset and feeds it to the model.
        The returned labels and loss will None if :attr:`return_loss` is False.

        Args:
            engine (internlm.core.Engine): InternLM engine for training and inference.
            data_iter (Iterable): Dataloader as the form of an iterator, obtained by calling iter(dataloader).
            forward_only (bool, optional):
                If True, the model is run for the forward pass, else back propagation will be executed.
            return_loss (bool, optional): Loss will be returned if True.
            return_output_label (bool, optional): Output and label will be returned if True.

        Returns:
            Tuple[:class:`torch.Tensor`]: A tuple of (output, label, loss), loss and label could be None.
        """
        assert (
            forward_only or return_loss
        ), "The argument 'return_loss' has to be True when 'forward_only' is False, but got False."

        batch_data, batch_size = engine.load_batch(data_iter)

        assert (
            batch_size % self._grad_accum_size == 0
        ), f"batch_size:{batch_size} must be an integer multiple of gradient accumulation steps:{self._grad_accum_size}"
        self._grad_accum_batch_size = batch_size // self._grad_accum_size

        data, label = batch_data

        loss = defaultdict(int) if return_loss else None
        outputs = []
        labels = []

        # reset accumulation microbatch offset
        self._grad_accum_offset = 0

        for _current_accum_step in range(self._grad_accum_size):
            if _current_accum_step == self._grad_accum_size - 1:
                engine.optimizer.skip_grad_reduce = False
            else:
                engine.optimizer.skip_grad_reduce = True

            _data, _label = self._load_accum_batch(data, label)

            _output, _loss = self._train_one_batch(
                _data, _label, engine, forward_only, return_loss, self._grad_accum_size
            )

            if return_loss:
                for k in _loss:
                    loss[k] += _loss[k]
            if return_output_label:
                outputs.append(_output)
                labels.append(_label)

        if not return_output_label:
            outputs, labels = None, None

        return outputs, labels, loss

class KDNonPipelineScheduler(NonPipelineScheduler):

    def __init__(
            self,
            data_process_func: Callable = None,
            gradient_accumulation_size: int = 1,
            scheduler_hooks: Optional[List[SchedulerHook]] = None,
    ):
        super().__init__(
            data_process_func=data_process_func,
            gradient_accumulation_size=gradient_accumulation_size,
            scheduler_hooks=scheduler_hooks,
        )

    def _train_one_batch(
            self,
            data: Any,
            label: Any,
            engine: KDEngine,
            forward_only: bool = False,
            return_loss: bool = True,
            scale_loss: int = 1,
    ):
        """Trains one batch of data.

        Args:
            data (Any): The data to be trained.
            label (Any): The label for the data.
            engine (internlm.core.Engine): InternLM engine for training and inference.
            forward_only (bool, optional): If True, the model is run for the forward pass, else back propagation will
                be executed.
            return_loss (bool, optional): Loss will be returned if True.
            scale_loss (int, optional): The scale factor for the loss.
        """

        # forward
        with conditional_context(torch.no_grad(), enable=forward_only):
            self._call_hooks("before_forward", data)
            output = self._call_engine(engine, data)
            self._call_hooks("after_forward", output)

            self._call_hooks("post_helper_func", output, label)

            if return_loss:
                self._call_hooks("before_criterion", output, label)
                loss_gt = gpc.config.kd_config['gt_weight'] * self._call_engine_criterion(engine.criterion, output, label)

                with torch.no_grad():
                    engine.teacher.eval()
                    output_t = self._call_engine(engine.teacher, data)

                loss_kd = gpc.config.kd_config['kd_weight'] * self._call_engine_criterion(engine.kd_criterion, output, (output_t, label))

                self._call_hooks("after_criterion", loss_gt + loss_kd)
                loss_gt /= scale_loss
                loss_kd /= scale_loss

        # backward
        if not forward_only:
            self._call_hooks("before_backward", None, None)
            engine.backward(loss_gt+loss_kd)
            self._call_hooks("after_backward", None)

        if not return_loss:
            loss_gt = None
            loss_kd = None

        return output, dict(loss_gt=loss_gt, loss_kd=loss_kd)


class KDNonPipelineSchedulerContrastive(NonPipelineScheduler):

    def __init__(
            self,
            data_process_func: Callable = None,
            gradient_accumulation_size: int = 1,
            scheduler_hooks: Optional[List[SchedulerHook]] = None,
            momentum_scheduler: Optional[np.ndarray] = None,
    ):
        super().__init__(
            data_process_func=data_process_func,
            gradient_accumulation_size=gradient_accumulation_size,
            scheduler_hooks=scheduler_hooks,
        )

        self.momentum_scheduler = momentum_scheduler

    def _train_one_batch(
            self,
            data: Any,
            label: Any,
            engine: KDEngine,
            forward_only: bool = False,
            return_loss: bool = True,
            scale_loss: int = 1,
            global_iteration: int = 1,
    ):
        """Trains one batch of data.

        Args:
            data (Any): The data to be trained.
            label (Any): The label for the data.
            engine (internlm.core.Engine): InternLM engine for training and inference.
            forward_only (bool, optional): If True, the model is run for the forward pass, else back propagation will
                be executed.
            return_loss (bool, optional): Loss will be returned if True.
            scale_loss (int, optional): The scale factor for the loss.
        """

        # forward
        with conditional_context(torch.no_grad(), enable=forward_only):
            self._call_hooks("before_forward", data)
            output = self._call_engine(engine, data) # engine.model (student)
            self._call_hooks("after_forward", output)

            self._call_hooks("post_helper_func", output, label)

            if return_loss:
                self._call_hooks("before_criterion", output, label)
                # loss_gt = gpc.config.kd_config['gt_weight'] * self._call_engine_criterion(engine.criterion, output, label)

                with torch.no_grad():
                    # engine.teacher.eval()
                    output_t = self._call_engine(engine.teacher, data) # (teacher)

                loss_kd = gpc.config.kd_config['kd_weight'] * self._call_engine_criterion(engine.kd_criterion, output, (output_t, label))

                self._call_hooks("after_criterion", loss_kd)
                # loss_gt /= scale_loss
                loss_kd /= scale_loss

        # backward
        if not forward_only:
            self._call_hooks("before_backward", None, None)
            # student update
            engine.backward(loss_kd)
            # teacher EMA update
            # Note: this is wrong, already moved out to train scripts
            # with torch.no_grad():
            #     m = self.momentum_scheduler[global_iteration]  # momentum parameter
            #     for (name_q, param_q), (name_k, param_k) in zip(engine.model.named_parameters(), engine.teacher.named_parameters()):
            #         # do not update the vqgan model
            #         if name_q.startswith("model.embedding.vq_model"):
            #             continue
            #         param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
            self._call_hooks("after_backward", None)

        if not return_loss:
            # loss_gt = None
            loss_kd = None

        return output, dict(loss_kd=loss_kd)
    
    @llm_timeout(func_name="nopp_forward_backward_step_contrastive")
    def forward_backward_step(
        self,
        engine: Engine,
        data_iter: Iterable,
        forward_only: bool = False,
        return_loss: bool = True,
        return_output_label: bool = True,
        global_iteration: int = 1,
    ):
        """The process function that loads a batch of dataset and feeds it to the model.
        The returned labels and loss will None if :attr:`return_loss` is False.

        Args:
            engine (internlm.core.Engine): InternLM engine for training and inference.
            data_iter (Iterable): Dataloader as the form of an iterator, obtained by calling iter(dataloader).
            forward_only (bool, optional):
                If True, the model is run for the forward pass, else back propagation will be executed.
            return_loss (bool, optional): Loss will be returned if True.
            return_output_label (bool, optional): Output and label will be returned if True.

        Returns:
            Tuple[:class:`torch.Tensor`]: A tuple of (output, label, loss), loss and label could be None.
        """
        assert (
            forward_only or return_loss
        ), "The argument 'return_loss' has to be True when 'forward_only' is False, but got False."

        batch_data, batch_size = engine.load_batch(data_iter)

        assert (
            batch_size % self._grad_accum_size == 0
        ), f"batch_size:{batch_size} must be an integer multiple of gradient accumulation steps:{self._grad_accum_size}"
        self._grad_accum_batch_size = batch_size // self._grad_accum_size

        data, label = batch_data

        loss = defaultdict(int) if return_loss else None
        outputs = []
        labels = []

        # reset accumulation microbatch offset
        self._grad_accum_offset = 0

        for _current_accum_step in range(self._grad_accum_size):
            if _current_accum_step == self._grad_accum_size - 1:
                engine.optimizer.skip_grad_reduce = False
            else:
                engine.optimizer.skip_grad_reduce = True

            _data, _label = self._load_accum_batch(data, label)

            _output, _loss = self._train_one_batch(
                _data, _label, engine, forward_only, return_loss, self._grad_accum_size, global_iteration,
            )

            if return_loss:
                for k in _loss:
                    loss[k] += _loss[k]
            if return_output_label:
                outputs.append(_output)
                labels.append(_label)

        if not return_output_label:
            outputs, labels = None, None

        return outputs, labels, loss


class KDNonPipelineSchedulerContrastive_GPT(NonPipelineScheduler):

    def __init__(
            self,
            data_process_func: Callable = None,
            gradient_accumulation_size: int = 1,
            scheduler_hooks: Optional[List[SchedulerHook]] = None,
            momentum_scheduler: Optional[np.ndarray] = None,
    ):
        super().__init__(
            data_process_func=data_process_func,
            gradient_accumulation_size=gradient_accumulation_size,
            scheduler_hooks=scheduler_hooks,
        )

        self.momentum_scheduler = momentum_scheduler

    def _train_one_batch(
            self,
            data: Any,
            label: Any,
            engine: KDEngine,
            forward_only: bool = False,
            return_loss: bool = True,
            scale_loss: int = 1,
            global_iteration: int = 1,
    ):
        """Trains one batch of data.

        Args:
            data (Any): The data to be trained.
            label (Any): The label for the data.
            engine (internlm.core.Engine): InternLM engine for training and inference.
            forward_only (bool, optional): If True, the model is run for the forward pass, else back propagation will
                be executed.
            return_loss (bool, optional): Loss will be returned if True.
            scale_loss (int, optional): The scale factor for the loss.
        """

        # forward
        with conditional_context(torch.no_grad(), enable=forward_only):
            self._call_hooks("before_forward", data)
            output = self._call_engine(engine, data)  # engine.model (student)
            self._call_hooks("after_forward", output)

            self._call_hooks("post_helper_func", output, label)

            if return_loss:
                self._call_hooks("before_criterion", output, label)
                loss_gt = gpc.config.kd_config['gt_weight'] * self._call_engine_criterion(engine.criterion, output, label)

                with torch.no_grad():
                    # engine.teacher.eval()
                    output_t = self._call_engine(engine.teacher, data)  # (teacher)
                # import pdb; pdb.set_trace()
                loss_kd = gpc.config.kd_config['kd_weight'] * self._call_engine_criterion(engine.kd_criterion, output,
                                                                                          output_t, )

                self._call_hooks("after_criterion", loss_kd)
                loss_gt /= scale_loss
                loss_kd /= scale_loss

                loss = loss_gt + loss_kd

        # backward
        if not forward_only:
            self._call_hooks("before_backward", None, None)
            # student update
            engine.backward(loss)
            # teacher EMA update
            # Note: this is wrong, already moved out to train scripts
            # with torch.no_grad():
            #     m = self.momentum_scheduler[global_iteration]  # momentum parameter
            #     for (name_q, param_q), (name_k, param_k) in zip(engine.model.named_parameters(), engine.teacher.named_parameters()):
            #         # do not update the vqgan model
            #         if name_q.startswith("model.embedding.vq_model"):
            #             continue
            #         param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
            self._call_hooks("after_backward", None)

        if not return_loss:
            loss_gt = None
            loss_kd = None

        return output, dict(loss_kd=loss_kd, loss_gt=loss_gt)

    @llm_timeout(func_name="nopp_forward_backward_step_contrastive")
    def forward_backward_step(
            self,
            engine: Engine,
            data_iter: Iterable,
            forward_only: bool = False,
            return_loss: bool = True,
            return_output_label: bool = True,
            global_iteration: int = 1,
    ):
        """The process function that loads a batch of dataset and feeds it to the model.
        The returned labels and loss will None if :attr:`return_loss` is False.

        Args:
            engine (internlm.core.Engine): InternLM engine for training and inference.
            data_iter (Iterable): Dataloader as the form of an iterator, obtained by calling iter(dataloader).
            forward_only (bool, optional):
                If True, the model is run for the forward pass, else back propagation will be executed.
            return_loss (bool, optional): Loss will be returned if True.
            return_output_label (bool, optional): Output and label will be returned if True.

        Returns:
            Tuple[:class:`torch.Tensor`]: A tuple of (output, label, loss), loss and label could be None.
        """
        assert (
                forward_only or return_loss
        ), "The argument 'return_loss' has to be True when 'forward_only' is False, but got False."

        batch_data, batch_size = engine.load_batch(data_iter)

        assert (
                batch_size % self._grad_accum_size == 0
        ), f"batch_size:{batch_size} must be an integer multiple of gradient accumulation steps:{self._grad_accum_size}"
        self._grad_accum_batch_size = batch_size // self._grad_accum_size

        data, label = batch_data

        loss = defaultdict(int) if return_loss else None
        outputs = []
        labels = []

        # reset accumulation microbatch offset
        self._grad_accum_offset = 0

        for _current_accum_step in range(self._grad_accum_size):
            if _current_accum_step == self._grad_accum_size - 1:
                engine.optimizer.skip_grad_reduce = False
            else:
                engine.optimizer.skip_grad_reduce = True

            _data, _label = self._load_accum_batch(data, label)

            _output, _loss = self._train_one_batch(
                _data, _label, engine, forward_only, return_loss, self._grad_accum_size, global_iteration,
            )

            if return_loss:
                for k in _loss:
                    loss[k] += _loss[k]
            if return_output_label:
                outputs.append(_output)
                labels.append(_label)

        if not return_output_label:
            outputs, labels = None, None

        return outputs, labels, loss

class KDNonPipelineSchedulerContrastive_motion_regression(NonPipelineScheduler):

    def __init__(
            self,
            data_process_func: Callable = None,
            gradient_accumulation_size: int = 1,
            scheduler_hooks: Optional[List[SchedulerHook]] = None,
            momentum_scheduler: Optional[np.ndarray] = None,
    ):
        super().__init__(
            data_process_func=data_process_func,
            gradient_accumulation_size=gradient_accumulation_size,
            scheduler_hooks=scheduler_hooks,
        )

        self.momentum_scheduler = momentum_scheduler

    def _train_one_batch(
            self,
            data: Any,
            label: Any,
            engine: KDEngine,
            forward_only: bool = False,
            return_loss: bool = True,
            scale_loss: int = 1,
            global_iteration: int = 1,
    ):
        """Trains one batch of data.

        Args:
            data (Any): The data to be trained.
            label (Any): The label for the data.
            engine (internlm.core.Engine): InternLM engine for training and inference.
            forward_only (bool, optional): If True, the model is run for the forward pass, else back propagation will
                be executed.
            return_loss (bool, optional): Loss will be returned if True.
            scale_loss (int, optional): The scale factor for the loss.
        """

        # forward
        with conditional_context(torch.no_grad(), enable=forward_only):
            self._call_hooks("before_forward", data)
            output = self._call_engine(engine, data)  # engine.model (student)
            self._call_hooks("after_forward", output)

            self._call_hooks("post_helper_func", output, label)

            if return_loss:
                self._call_hooks("before_criterion", output, label)
                loss_pose, loss_conf = self._call_engine_criterion(engine.criterion, output, label)

                # copy the loss for logging
                loss_pose_raw = loss_pose.detach()
                loss_conf_raw = loss_conf.detach()

                loss_pose = gpc.config.kd_config['reg_weight'] * loss_pose

                with torch.no_grad():
                    # engine.teacher.eval()
                    output_t = self._call_engine(engine.teacher, data)  # (teacher)

                loss_kd = self._call_engine_criterion(engine.kd_criterion, output, output_t, )
                loss_kd_raw = loss_kd.detach()

                loss_kd = gpc.config.kd_config['kd_weight'] * loss_kd

                self._call_hooks("after_criterion", loss_kd)
                loss_pose /= scale_loss
                loss_kd /= scale_loss
                loss_conf /= scale_loss

                loss_kd_raw /= scale_loss
                loss_pose_raw /= scale_loss
                loss_conf_raw /= scale_loss


                loss_gt = loss_pose + loss_conf
                loss_gt = gpc.config.kd_config['gt_weight'] * loss_gt * self.momentum_scheduler[global_iteration]

                loss = loss_gt + loss_kd

                if torch.isnan(loss):
                    import pdb; pdb.set_trace()

                # loss_gt_raw = loss_pose + loss_conf


        # backward
        if not forward_only:
            self._call_hooks("before_backward", None, None)
            # student update
            engine.backward(loss)

            self._call_hooks("after_backward", None)

        if not return_loss:
            loss_gt = None
            loss_kd = None
            loss_pose, loss_conf = None, None
            loss = None
            loss_conf_raw = None
            loss_pose_raw = None
            loss_kd_raw = None
        return output, dict(loss=loss, loss_kd=loss_kd_raw, loss_gt=loss_gt, loss_pose=loss_pose_raw, loss_conf=loss_conf_raw)

    @llm_timeout(func_name="nopp_forward_backward_step_contrastive")
    def forward_backward_step(
            self,
            engine: Engine,
            data_iter: Iterable,
            forward_only: bool = False,
            return_loss: bool = True,
            return_output_label: bool = True,
            global_iteration: int = 1,
    ):
        """The process function that loads a batch of dataset and feeds it to the model.
        The returned labels and loss will None if :attr:`return_loss` is False.

        Args:
            engine (internlm.core.Engine): InternLM engine for training and inference.
            data_iter (Iterable): Dataloader as the form of an iterator, obtained by calling iter(dataloader).
            forward_only (bool, optional):
                If True, the model is run for the forward pass, else back propagation will be executed.
            return_loss (bool, optional): Loss will be returned if True.
            return_output_label (bool, optional): Output and label will be returned if True.

        Returns:
            Tuple[:class:`torch.Tensor`]: A tuple of (output, label, loss), loss and label could be None.
        """
        assert (
                forward_only or return_loss
        ), "The argument 'return_loss' has to be True when 'forward_only' is False, but got False."

        batch_data, batch_size = engine.load_batch(data_iter)

        assert (
                batch_size % self._grad_accum_size == 0
        ), f"batch_size:{batch_size} must be an integer multiple of gradient accumulation steps:{self._grad_accum_size}"
        self._grad_accum_batch_size = batch_size // self._grad_accum_size

        data, label = batch_data

        loss = defaultdict(int) if return_loss else None
        outputs = []
        labels = []

        # reset accumulation microbatch offset
        self._grad_accum_offset = 0

        for _current_accum_step in range(self._grad_accum_size):
            if _current_accum_step == self._grad_accum_size - 1:
                engine.optimizer.skip_grad_reduce = False
            else:
                engine.optimizer.skip_grad_reduce = True

            _data, _label = self._load_accum_batch(data, label)

            _output, _loss = self._train_one_batch(
                _data, _label, engine, forward_only, return_loss, self._grad_accum_size, global_iteration,
            )

            if return_loss:
                for k in _loss:
                    loss[k] += _loss[k]
            if return_output_label:
                outputs.append(_output)
                labels.append(_label)

        if not return_output_label:
            outputs, labels = None, None

        return outputs, labels, loss


class KDNonPipelineSchedulerContrastive_motion_regression_dino(NonPipelineScheduler):

    def __init__(
            self,
            data_process_func: Callable = None,
            gradient_accumulation_size: int = 1,
            scheduler_hooks: Optional[List[SchedulerHook]] = None,
            momentum_scheduler: Optional[np.ndarray] = None,
    ):
        super().__init__(
            data_process_func=data_process_func,
            gradient_accumulation_size=gradient_accumulation_size,
            scheduler_hooks=scheduler_hooks,
        )

        self.momentum_scheduler = momentum_scheduler

    def _train_img_and_motion(self, engine, label, data, scale_loss, global_iteration):
        # import pdb;pdb.set_trace()
        self._call_hooks("before_forward", data)
        output = self._call_engine(engine, data)  # engine.model (student)
        self._call_hooks("after_forward", output)

        self._call_hooks("post_helper_func", output, label)

        self._call_hooks("before_criterion", output, label)
        loss_pose, loss_conf = self._call_engine_criterion(engine.criterion, output, label)

        # copy the loss for logging
        loss_pose_raw = loss_pose.detach()
        loss_conf_raw = loss_conf.detach()

        loss_pose = gpc.config.kd_config['reg_weight'] * loss_pose

        with torch.no_grad():
            # engine.teacher.eval()
            output_t = self._call_engine(engine.teacher, data)  # (teacher)

        loss_kd = self._call_engine_criterion(engine.kd_criterion[0], output, output_t, )
        loss_kd_raw = loss_kd.detach()

        loss_kd = gpc.config.kd_config['kd_weight'] * loss_kd

        self._call_hooks("after_criterion", loss_kd)
        loss_pose /= scale_loss
        loss_kd /= scale_loss
        loss_conf /= scale_loss

        loss_kd_raw /= scale_loss
        loss_pose_raw /= scale_loss
        loss_conf_raw /= scale_loss

        loss_gt = loss_pose + loss_conf
        loss_gt = gpc.config.kd_config['gt_weight'] * loss_gt * self.momentum_scheduler[global_iteration]

        loss = loss_gt + loss_kd

        if torch.isnan(loss):
            import pdb;
            pdb.set_trace()

        return loss, loss_gt, loss_kd_raw, loss_pose_raw, loss_conf_raw, output

    def _train_dino(self, engine, label, data, scale_loss, global_iteration):
        # import pdb;pdb.set_trace()
        self._call_hooks("before_forward", data)
        output = self._call_engine(engine, data)  # engine.model (student)
        self._call_hooks("after_forward", output)

        self._call_hooks("post_helper_func", output, label)


        self._call_hooks("before_criterion", output, label)
        # loss_gt = gpc.config.kd_config['gt_weight'] * self._call_engine_criterion(engine.criterion, output, label)

        with torch.no_grad():
            # engine.teacher.eval()
            output_t = self._call_engine(engine.teacher, data)  # (teacher)

        loss_dino_raw = self._call_engine_criterion(engine.kd_criterion[1], output, (output_t, label))

        loss_dino = gpc.config.kd_config['dino_weight'] * loss_dino_raw

        self._call_hooks("after_criterion", loss_dino)
        # loss_gt /= scale_loss
        loss_dino /= scale_loss

        return loss_dino, loss_dino_raw, output

    def _train_one_batch(
            self,
            data: Any,
            label: Any,
            engine: KDEngine,
            forward_only: bool = False,
            return_loss: bool = True,
            scale_loss: int = 1,
            global_iteration: int = 1,
    ):
        """Trains one batch of data.

        Args:
            data (Any): The data to be trained.
            label (Any): The label for the data.
            engine (internlm.core.Engine): InternLM engine for training and inference.
            forward_only (bool, optional): If True, the model is run for the forward pass, else back propagation will
                be executed.
            return_loss (bool, optional): Loss will be returned if True.
            scale_loss (int, optional): The scale factor for the loss.
        """
        assert type(engine.kd_criterion)==list

        # forward
        with (conditional_context(torch.no_grad(), enable=forward_only)):

            if 'poses' in data.keys():
                loss_img_motion, loss_gt, loss_kd_raw, loss_pose_raw, loss_conf_raw, output = \
                self._train_img_and_motion(engine, label, data, scale_loss, global_iteration)
                loss_dino = None
                loss_dino_raw = None
                loss = loss_img_motion
            else:
                loss_dino, loss_dino_raw, output = self._train_dino(engine, label, data, scale_loss, global_iteration)
                loss, loss_gt, loss_kd_raw, loss_pose_raw, loss_conf_raw = loss_dino, None, None, None, None
                loss_img_motion = None
        # backward
        if not forward_only:
            self._call_hooks("before_backward", None, None)
            # student update
            engine.backward(loss)

            self._call_hooks("after_backward", None)

        if not return_loss:
            loss_gt = None
            loss = None
            loss_img_motion = None
            loss_conf_raw = None
            loss_pose_raw = None
            loss_kd_raw = None
            loss_dino = None
            loss_dino_raw = None
        return output, dict(loss=loss, loss_kd=loss_kd_raw, loss_gt=loss_gt, loss_img_motion=loss_img_motion,
                            loss_pose=loss_pose_raw, loss_conf=loss_conf_raw, loss_dino=loss_dino, loss_dino_raw=loss_dino_raw)

    @llm_timeout(func_name="nopp_forward_backward_step_contrastive")
    def forward_backward_step(
            self,
            engine: Engine,
            data_iter: Iterable,
            forward_only: bool = False,
            return_loss: bool = True,
            return_output_label: bool = True,
            global_iteration: int = 1,
    ):
        """The process function that loads a batch of dataset and feeds it to the model.
        The returned labels and loss will None if :attr:`return_loss` is False.

        Args:
            engine (internlm.core.Engine): InternLM engine for training and inference.
            data_iter (Iterable): Dataloader as the form of an iterator, obtained by calling iter(dataloader).
            forward_only (bool, optional):
                If True, the model is run for the forward pass, else back propagation will be executed.
            return_loss (bool, optional): Loss will be returned if True.
            return_output_label (bool, optional): Output and label will be returned if True.

        Returns:
            Tuple[:class:`torch.Tensor`]: A tuple of (output, label, loss), loss and label could be None.
        """
        assert (
                forward_only or return_loss
        ), "The argument 'return_loss' has to be True when 'forward_only' is False, but got False."

        batch_data, batch_size = engine.load_batch(data_iter)

        assert (
                batch_size % self._grad_accum_size == 0
        ), f"batch_size:{batch_size} must be an integer multiple of gradient accumulation steps:{self._grad_accum_size}"
        self._grad_accum_batch_size = batch_size // self._grad_accum_size

        data, label = batch_data

        loss = defaultdict(int) if return_loss else None
        outputs = []
        labels = []

        # reset accumulation microbatch offset
        self._grad_accum_offset = 0

        for _current_accum_step in range(self._grad_accum_size):
            if _current_accum_step == self._grad_accum_size - 1:
                engine.optimizer.skip_grad_reduce = False
            else:
                engine.optimizer.skip_grad_reduce = True

            _data, _label = self._load_accum_batch(data, label)

            _output, _loss = self._train_one_batch(
                _data, _label, engine, forward_only, return_loss, self._grad_accum_size, global_iteration,
            )

            if return_loss:
                for k in _loss:
                    if _loss[k] is not None:
                        loss[k] += _loss[k]
            if return_output_label:
                outputs.append(_output)
                labels.append(_label)

        if not return_output_label:
            outputs, labels = None, None

        return outputs, labels, loss