# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" OpenAI GPT-2 configuration """

import os
import copy
import json
from typing import Any, Dict, Tuple, Union

class PretrainedConfig():
    r"""
    Base class for all configuration classes. Handles a few parameters common to all models' configurations as well as
    methods for loading/downloading/saving configurations.

    Note:
        A configuration file can be loaded and saved to disk. Loading the configuration file and using this file to
        initialize a model does **not** load the model weights. It only affects the model's configuration.

    Class attributes (overridden by derived classes)

        - **model_type** (:obj:`str`) -- An identifier for the model type, serialized into the JSON file, and used to
          recreate the correct object in :class:`~transformers.AutoConfig`.
        - **is_composition** (:obj:`bool`) -- Whether the config class is composed of multiple sub-configs. In this
          case the config has to be initialized from two or more configs of type
          :class:`~transformers.PretrainedConfig` like: :class:`~transformers.EncoderDecoderConfig` or
          :class:`~RagConfig`.
        - **keys_to_ignore_at_inference** (:obj:`List[str]`) -- A list of keys to ignore by default when looking at
          dictionary outputs of the model during inference.

    Common attributes (present in all subclasses)

        - **vocab_size** (:obj:`int`) -- The number of tokens in the vocabulary, which is also the first dimension of
          the embeddings matrix (this attribute may be missing for models that don't have a text modality like ViT).
        - **hidden_size** (:obj:`int`) -- The hidden size of the model.
        - **num_attention_heads** (:obj:`int`) -- The number of attention heads used in the multi-head attention layers
          of the model.
        - **num_hidden_layers** (:obj:`int`) -- The number of blocks in the model.

    Args:
        name_or_path (:obj:`str`, `optional`, defaults to :obj:`""`):
            Store the string that was passed to :func:`~transformers.PreTrainedModel.from_pretrained` or
            :func:`~transformers.TFPreTrainedModel.from_pretrained` as ``pretrained_model_name_or_path`` if the
            configuration was created with such a method.
        output_hidden_states (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the model should return all hidden-states.
        output_attentions (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the model should returns all attentions.
        return_dict (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return a :class:`~transformers.file_utils.ModelOutput` instead of a plain
            tuple.
        is_encoder_decoder (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether the model is used as an encoder/decoder or not.
        is_decoder (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether the model is used as decoder or not (in which case it's used as an encoder).
        add_cross_attention (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether cross-attention layers should be added to the model. Note, this option is only relevant for models
            that can be used as decoder models within the `:class:~transformers.EncoderDecoderModel` class, which
            consists of all models in ``AUTO_MODELS_FOR_CAUSAL_LM``.
        tie_encoder_decoder (:obj:`bool`, `optional`, defaults to :obj:`False`)
            Whether all encoder weights should be tied to their equivalent decoder weights. This requires the encoder
            and decoder model to have the exact same parameter names.
        prune_heads (:obj:`Dict[int, List[int]]`, `optional`, defaults to :obj:`{}`):
            Pruned heads of the model. The keys are the selected layer indices and the associated values, the list of
            heads to prune in said layer.

            For instance ``{1: [0, 2], 2: [2, 3]}`` will prune heads 0 and 2 on layer 1 and heads 2 and 3 on layer 2.
        chunk_size_feed_forward (:obj:`int`, `optional`, defaults to :obj:`0`):
            The chunk size of all feed forward layers in the residual attention blocks. A chunk size of :obj:`0` means
            that the feed forward layer is not chunked. A chunk size of n means that the feed forward layer processes
            :obj:`n` < sequence_length embeddings at a time. For more information on feed forward chunking, see `How
            does Feed Forward Chunking work? <../glossary.html#feed-forward-chunking>`__ .

    Parameters for sequence generation

        - **max_length** (:obj:`int`, `optional`, defaults to 20) -- Maximum length that will be used by default in the
          :obj:`generate` method of the model.
        - **min_length** (:obj:`int`, `optional`, defaults to 10) -- Minimum length that will be used by default in the
          :obj:`generate` method of the model.
        - **do_sample** (:obj:`bool`, `optional`, defaults to :obj:`False`) -- Flag that will be used by default in the
          :obj:`generate` method of the model. Whether or not to use sampling ; use greedy decoding otherwise.
        - **early_stopping** (:obj:`bool`, `optional`, defaults to :obj:`False`) -- Flag that will be used by default
          in the :obj:`generate` method of the model. Whether to stop the beam search when at least ``num_beams``
          sentences are finished per batch or not.
        - **num_beams** (:obj:`int`, `optional`, defaults to 1) -- Number of beams for beam search that will be used by
          default in the :obj:`generate` method of the model. 1 means no beam search.
        - **num_beam_groups** (:obj:`int`, `optional`, defaults to 1) -- Number of groups to divide :obj:`num_beams`
          into in order to ensure diversity among different groups of beams that will be used by default in the
          :obj:`generate` method of the model. 1 means no group beam search.
        - **diversity_penalty** (:obj:`float`, `optional`, defaults to 0.0) -- Value to control diversity for group
          beam search. that will be used by default in the :obj:`generate` method of the model. 0 means no diversity
          penalty. The higher the penalty, the more diverse are the outputs.
        - **temperature** (:obj:`float`, `optional`, defaults to 1) -- The value used to module the next token
          probabilities that will be used by default in the :obj:`generate` method of the model. Must be strictly
          positive.
        - **top_k** (:obj:`int`, `optional`, defaults to 50) -- Number of highest probability vocabulary tokens to keep
          for top-k-filtering that will be used by default in the :obj:`generate` method of the model.
        - **top_p** (:obj:`float`, `optional`, defaults to 1) -- Value that will be used by default in the
          :obj:`generate` method of the model for ``top_p``. If set to float < 1, only the most probable tokens with
          probabilities that add up to ``top_p`` or higher are kept for generation.
        - **repetition_penalty** (:obj:`float`, `optional`, defaults to 1) -- Parameter for repetition penalty that
          will be used by default in the :obj:`generate` method of the model. 1.0 means no penalty.
        - **length_penalty** (:obj:`float`, `optional`, defaults to 1) -- Exponential penalty to the length that will
          be used by default in the :obj:`generate` method of the model.
        - **no_repeat_ngram_size** (:obj:`int`, `optional`, defaults to 0) -- Value that will be used by default in the
          :obj:`generate` method of the model for ``no_repeat_ngram_size``. If set to int > 0, all ngrams of that size
          can only occur once.
        - **encoder_no_repeat_ngram_size** (:obj:`int`, `optional`, defaults to 0) -- Value that will be used by
          default in the :obj:`generate` method of the model for ``encoder_no_repeat_ngram_size``. If set to int > 0,
          all ngrams of that size that occur in the ``encoder_input_ids`` cannot occur in the ``decoder_input_ids``.
        - **bad_words_ids** (:obj:`List[int]`, `optional`) -- List of token ids that are not allowed to be generated
          that will be used by default in the :obj:`generate` method of the model. In order to get the tokens of the
          words that should not appear in the generated text, use :obj:`tokenizer.encode(bad_word,
          add_prefix_space=True)`.
        - **num_return_sequences** (:obj:`int`, `optional`, defaults to 1) -- Number of independently computed returned
          sequences for each element in the batch that will be used by default in the :obj:`generate` method of the
          model.
        - **output_scores** (:obj:`bool`, `optional`, defaults to :obj:`False`) -- Whether the model should return the
          logits when used for generation
        - **return_dict_in_generate** (:obj:`bool`, `optional`, defaults to :obj:`False`) -- Whether the model should
          return a :class:`~transformers.file_utils.ModelOutput` instead of a :obj:`torch.LongTensor`
        - **forced_bos_token_id** (:obj:`int`, `optional`) -- The id of the token to force as the first generated token
          after the :obj:`decoder_start_token_id`. Useful for multilingual models like :doc:`mBART
          <../model_doc/mbart>` where the first generated token needs to be the target language token.
        - **forced_eos_token_id** (:obj:`int`, `optional`) -- The id of the token to force as the last generated token
          when :obj:`max_length` is reached.
        - **remove_invalid_values** (:obj:`bool`, `optional`) -- Whether to remove possible `nan` and `inf` outputs of
          the model to prevent the generation method to crash. Note that using ``remove_invalid_values`` can slow down
          generation.


    Parameters for fine-tuning tasks

        - **architectures** (:obj:`List[str]`, `optional`) -- Model architectures that can be used with the model
          pretrained weights.
        - **finetuning_task** (:obj:`str`, `optional`) -- Name of the task used to fine-tune the model. This can be
          used when converting from an original (TensorFlow or PyTorch) checkpoint.
        - **id2label** (:obj:`Dict[int, str]`, `optional`) -- A map from index (for instance prediction index, or
          target index) to label.
        - **label2id** (:obj:`Dict[str, int]`, `optional`) -- A map from label to index for the model.
        - **num_labels** (:obj:`int`, `optional`) -- Number of labels to use in the last layer added to the model,
          typically for a classification task.
        - **task_specific_params** (:obj:`Dict[str, Any]`, `optional`) -- Additional keyword arguments to store for the
          current task.
        - **problem_type** (:obj:`str`, `optional`) -- Problem type for :obj:`XxxForSequenceClassification` models. Can
          be one of (:obj:`"regression"`, :obj:`"single_label_classification"`, :obj:`"multi_label_classification"`).
          Please note that this parameter is only available in the following models: `AlbertForSequenceClassification`,
          `BertForSequenceClassification`, `BigBirdForSequenceClassification`, `ConvBertForSequenceClassification`,
          `DistilBertForSequenceClassification`, `ElectraForSequenceClassification`, `FunnelForSequenceClassification`,
          `LongformerForSequenceClassification`, `MobileBertForSequenceClassification`,
          `ReformerForSequenceClassification`, `RobertaForSequenceClassification`,
          `SqueezeBertForSequenceClassification`, `XLMForSequenceClassification` and `XLNetForSequenceClassification`.

    Parameters linked to the tokenizer

        - **tokenizer_class** (:obj:`str`, `optional`) -- The name of the associated tokenizer class to use (if none is
          set, will use the tokenizer associated to the model by default).
        - **prefix** (:obj:`str`, `optional`) -- A specific prompt that should be added at the beginning of each text
          before calling the model.
        - **bos_token_id** (:obj:`int`, `optional`)) -- The id of the `beginning-of-stream` token.
        - **pad_token_id** (:obj:`int`, `optional`)) -- The id of the `padding` token.
        - **eos_token_id** (:obj:`int`, `optional`)) -- The id of the `end-of-stream` token.
        - **decoder_start_token_id** (:obj:`int`, `optional`)) -- If an encoder-decoder model starts decoding with a
          different token than `bos`, the id of that token.
        - **sep_token_id** (:obj:`int`, `optional`)) -- The id of the `separation` token.

    PyTorch specific parameters

        - **torchscript** (:obj:`bool`, `optional`, defaults to :obj:`False`) -- Whether or not the model should be
          used with Torchscript.
        - **tie_word_embeddings** (:obj:`bool`, `optional`, defaults to :obj:`True`) -- Whether the model's input and
          output word embeddings should be tied. Note that this is only relevant if the model has a output word
          embedding layer.
        - **torch_dtype** (:obj:`str`, `optional`) -- The :obj:`dtype` of the weights. This attribute can be used to
          initialize the model to a non-default ``dtype`` (which is normally ``float32``) and thus allow for optimal
          storage allocation. For example, if the saved model is ``float16``, ideally we want to load it back using the
          minimal amount of memory needed to load ``float16`` weights. Since the config object is stored in plain text,
          this attribute contains just the floating type string without the ``torch.`` prefix. For example, for
          ``torch.float16`` ``torch_dtype`` is the ``"float16"`` string.

    TensorFlow specific parameters

        - **use_bfloat16** (:obj:`bool`, `optional`, defaults to :obj:`False`) -- Whether or not the model should use
          BFloat16 scalars (only used by some TensorFlow models).
    """
    model_type: str = ""
    is_composition: bool = False

    def __init__(self, **kwargs):
        # Attributes with defaults
        self.return_dict = kwargs.pop("return_dict", True)
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)
        self.output_attentions = kwargs.pop("output_attentions", False)
        self.torchscript = kwargs.pop("torchscript", False)  # Only used by PyTorch models
        self.torch_dtype = kwargs.pop("torch_dtype", None)  # Only used by PyTorch models
        self.use_bfloat16 = kwargs.pop("use_bfloat16", False)
        self.pruned_heads = kwargs.pop("pruned_heads", {})
        self.tie_word_embeddings = kwargs.pop(
            "tie_word_embeddings", True
        )  # Whether input and output word embeddings should be tied for all MLM, LM and Seq2Seq models.

        # Is decoder is used in encoder-decoder models to differentiate encoder from decoder
        self.is_encoder_decoder = kwargs.pop("is_encoder_decoder", False)
        self.is_decoder = kwargs.pop("is_decoder", False)
        self.add_cross_attention = kwargs.pop("add_cross_attention", False)
        self.tie_encoder_decoder = kwargs.pop("tie_encoder_decoder", False)

        # Parameters for sequence generation
        self.max_length = kwargs.pop("max_length", 20)
        self.min_length = kwargs.pop("min_length", 0)
        self.do_sample = kwargs.pop("do_sample", False)
        self.early_stopping = kwargs.pop("early_stopping", False)
        self.num_beams = kwargs.pop("num_beams", 1)
        self.num_beam_groups = kwargs.pop("num_beam_groups", 1)
        self.diversity_penalty = kwargs.pop("diversity_penalty", 0.0)
        self.temperature = kwargs.pop("temperature", 1.0)
        self.top_k = kwargs.pop("top_k", 50)
        self.top_p = kwargs.pop("top_p", 1.0)
        self.repetition_penalty = kwargs.pop("repetition_penalty", 1.0)
        self.length_penalty = kwargs.pop("length_penalty", 1.0)
        self.no_repeat_ngram_size = kwargs.pop("no_repeat_ngram_size", 0)
        self.encoder_no_repeat_ngram_size = kwargs.pop("encoder_no_repeat_ngram_size", 0)
        self.bad_words_ids = kwargs.pop("bad_words_ids", None)
        self.num_return_sequences = kwargs.pop("num_return_sequences", 1)
        self.chunk_size_feed_forward = kwargs.pop("chunk_size_feed_forward", 0)
        self.output_scores = kwargs.pop("output_scores", False)
        self.return_dict_in_generate = kwargs.pop("return_dict_in_generate", False)
        self.forced_bos_token_id = kwargs.pop("forced_bos_token_id", None)
        self.forced_eos_token_id = kwargs.pop("forced_eos_token_id", None)
        self.remove_invalid_values = kwargs.pop("remove_invalid_values", False)
        self.conditional = kwargs.pop("conditional", False)

        # Fine-tuning task arguments
        self.architectures = kwargs.pop("architectures", None)
        self.finetuning_task = kwargs.pop("finetuning_task", None)
        self.id2label = kwargs.pop("id2label", None)
        self.label2id = kwargs.pop("label2id", None)
        if self.id2label is not None:
            kwargs.pop("num_labels", None)
            self.id2label = dict((int(key), value) for key, value in self.id2label.items())
            # Keys are always strings in JSON so convert ids to int here.
        else:
            self.num_labels = kwargs.pop("num_labels", 2)

        # Tokenizer arguments TODO: eventually tokenizer and models should share the same config
        self.tokenizer_class = kwargs.pop("tokenizer_class", None)
        self.prefix = kwargs.pop("prefix", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)
        self.sep_token_id = kwargs.pop("sep_token_id", None)

        self.decoder_start_token_id = kwargs.pop("decoder_start_token_id", None)

        # task specific arguments
        self.task_specific_params = kwargs.pop("task_specific_params", None)

        # regression / multi-label classification
        self.problem_type = kwargs.pop("problem_type", None)
        allowed_problem_types = ("regression", "single_label_classification", "multi_label_classification")
        if self.problem_type is not None and self.problem_type not in allowed_problem_types:
            raise ValueError(
                f"The config parameter `problem_type` wasnot understood: received {self.problem_type}"
                "but only 'regression', 'single_label_classification' and 'multi_label_classification' are valid."
            )

        # TPU arguments
        if kwargs.pop("xla_device", None) is not None:
            print(
                "The `xla_device` argument has been deprecated in v4.4.0 of Transformers. It is ignored and you can "
                "safely remove it from your `config.json` file."
            )

        # Name or path to the pretrained checkpoint
        self._name_or_path = str(kwargs.pop("name_or_path", ""))

        # Drop the transformers version info
        self.transformers_version = kwargs.pop("transformers_version", None)

        # Additional attributes without default values
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                raise RuntimeError(f"Can't set {key} with value {value} for {self}")

    @property
    def name_or_path(self) -> str:
        return self._name_or_path

    @name_or_path.setter
    def name_or_path(self, value):
        self._name_or_path = str(value)  # Make sure that name_or_path is a string (for JSON encoding)

    @property
    def use_return_dict(self) -> bool:
        """
        :obj:`bool`: Whether or not return :class:`~transformers.file_utils.ModelOutput` instead of tuples.
        """
        # If torchscript is set, force `return_dict=False` to avoid jit errors
        return self.return_dict and not self.torchscript

    @property
    def num_labels(self) -> int:
        """
        :obj:`int`: The number of labels for classification models.
        """
        return len(self.id2label)

    @num_labels.setter
    def num_labels(self, num_labels: int):
        if self.id2label is None or len(self.id2label) != num_labels:
            self.id2label = {i: f"LABEL_{i}" for i in range(num_labels)}
            self.label2id = dict(zip(self.id2label.values(), self.id2label.keys()))

    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        """
        Save a configuration object to the directory ``save_directory``, so that it can be re-loaded using the
        :func:`~transformers.PretrainedConfig.from_pretrained` class method.

        Args:
            save_directory (:obj:`str` or :obj:`os.PathLike`):
                Directory where the configuration JSON file will be saved (will be created if it does not exist).
            push_to_hub (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to push your model to the Hugging Face model hub after saving it.

                .. warning::

                    Using :obj:`push_to_hub=True` will synchronize the repository you are pushing to with
                    :obj:`save_directory`, which requires :obj:`save_directory` to be a local clone of the repo you are
                    pushing to if it's an existing folder. Pass along :obj:`temp_dir=True` to use a temporary directory
                    instead.

            kwargs:
                Additional key word arguments passed along to the
                :meth:`~transformers.file_utils.PushToHubMixin.push_to_hub` method.
        """
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        if push_to_hub:
            raise RuntimeError('Push to hub not supported.')

        os.makedirs(save_directory, exist_ok=True)
        # If we save using the predefined names, we can load using `from_pretrained`
        output_config_file = os.path.join(save_directory, 'config.json')

        self.to_json_file(output_config_file, use_diff=True)
        print(f"Configuration saved in {output_config_file}")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        r"""
        Instantiate a :class:`~transformers.PretrainedConfig` (or a derived class) from a pretrained model
        configuration.

        Args:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
                This can be either:

                - a string, the `model id` of a pretrained model configuration hosted inside a model repo on
                  huggingface.co. Valid model ids can be located at the root-level, like ``bert-base-uncased``, or
                  namespaced under a user or organization name, like ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing a configuration file saved using the
                  :func:`~transformers.PretrainedConfig.save_pretrained` method, e.g., ``./my_model_directory/``.
                - a path or url to a saved configuration JSON `file`, e.g.,
                  ``./my_model_directory/configuration.json``.
            cache_dir (:obj:`str` or :obj:`os.PathLike`, `optional`):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to force to (re-)download the configuration files and override the cached versions if
                they exist.
            resume_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to delete incompletely received file. Attempts to resume the download if such a file
                exists.
            proxies (:obj:`Dict[str, str]`, `optional`):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., :obj:`{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
            use_auth_token (:obj:`str` or `bool`, `optional`):
                The token to use as HTTP bearer authorization for remote files. If :obj:`True`, will use the token
                generated when running :obj:`transformers-cli login` (stored in :obj:`~/.huggingface`).
            revision(:obj:`str`, `optional`, defaults to :obj:`"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so ``revision`` can be any
                identifier allowed by git.
            return_unused_kwargs (:obj:`bool`, `optional`, defaults to :obj:`False`):
                If :obj:`False`, then this function returns just the final configuration object.

                If :obj:`True`, then this functions returns a :obj:`Tuple(config, unused_kwargs)` where `unused_kwargs`
                is a dictionary consisting of the key/value pairs whose keys are not configuration attributes: i.e.,
                the part of ``kwargs`` which has not been used to update ``config`` and is otherwise ignored.
            kwargs (:obj:`Dict[str, Any]`, `optional`):
                The values in kwargs of any keys which are configuration attributes will be used to override the loaded
                values. Behavior concerning key/value pairs whose keys are *not* configuration attributes is controlled
                by the ``return_unused_kwargs`` keyword parameter.

        .. note::

            Passing :obj:`use_auth_token=True` is required when you want to use a private model.


        Returns:
            :class:`PretrainedConfig`: The configuration object instantiated from this pretrained model.

        Examples::

            # We can't instantiate directly the base class `PretrainedConfig` so let's show the examples on a
            # derived class: BertConfig
            config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from huggingface.co and cache.
            config = BertConfig.from_pretrained('./test/saved_model/')  # E.g. config (or model) was saved using `save_pretrained('./test/saved_model/')`
            config = BertConfig.from_pretrained('./test/saved_model/my_configuration.json')
            config = BertConfig.from_pretrained('bert-base-uncased', output_attentions=True, foo=False)
            assert config.output_attentions == True
            config, unused_kwargs = BertConfig.from_pretrained('bert-base-uncased', output_attentions=True,
                                                               foo=False, return_unused_kwargs=True)
            assert config.output_attentions == True
            assert unused_kwargs == {'foo': False}

        """
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            print(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)

    @classmethod
    def get_config_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        From a ``pretrained_model_name_or_path``, resolve to a dictionary of parameters, to be used for instantiating a
        :class:`~transformers.PretrainedConfig` using ``from_dict``.



        Parameters:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.

        Returns:
            :obj:`Tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the configuration object.

        """
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        use_auth_token = kwargs.pop("use_auth_token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)

        user_agent = {"file_type": "config", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        # if is_offline_mode() and not local_files_only:
        #     print("Offline mode: forcing local_files_only=True")
        #     local_files_only = True

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        if os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path, 'config.json')
        elif os.path.isfile(pretrained_model_name_or_path):
            config_file = pretrained_model_name_or_path
        else:
            raise RuntimeError(f'Cannot find {pretrained_model_name_or_path}.')

        try:
            # # Load from URL or cache if already cached
            # resolved_config_file = cached_path(
            #     config_file,
            #     cache_dir=cache_dir,
            #     force_download=force_download,
            #     proxies=proxies,
            #     resume_download=resume_download,
            #     local_files_only=local_files_only,
            #     use_auth_token=use_auth_token,
            #     user_agent=user_agent,
            # )
            # Load config dict
            config_dict = cls._dict_from_json_file(config_file)

        except json.JSONDecodeError:
            msg = (
                f"Configuration file is not a valid JSON file. "
                f"Please check file content here: {config_file}."
            )
            raise EnvironmentError(msg)

        print(f"loading configuration file {config_file}")

        return config_dict, kwargs

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> "PretrainedConfig":
        """
        Instantiates a :class:`~transformers.PretrainedConfig` from a Python dictionary of parameters.

        Args:
            config_dict (:obj:`Dict[str, Any]`):
                Dictionary that will be used to instantiate the configuration object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the
                :func:`~transformers.PretrainedConfig.get_config_dict` method.
            kwargs (:obj:`Dict[str, Any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            :class:`PretrainedConfig`: The configuration object instantiated from those parameters.
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        config = cls(**config_dict)

        if hasattr(config, "pruned_heads"):
            config.pruned_heads = dict((int(key), value) for key, value in config.pruned_heads.items())

        # Update config with kwargs if needed
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        print(f"Model config {config}")
        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike]) -> "PretrainedConfig":
        """
        Instantiates a :class:`~transformers.PretrainedConfig` from the path to a JSON file of parameters.

        Args:
            json_file (:obj:`str` or :obj:`os.PathLike`):
                Path to the JSON file containing the parameters.

        Returns:
            :class:`PretrainedConfig`: The configuration object instantiated from that JSON file.

        """
        config_dict = cls._dict_from_json_file(json_file)
        return cls(**config_dict)

    @classmethod
    def _dict_from_json_file(cls, json_file: Union[str, os.PathLike]):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    def __eq__(self, other):
        return isinstance(other, PretrainedConfig) and self.__dict__ == other.__dict__

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def to_diff_dict(self) -> Dict[str, Any]:
        """
        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.

        Returns:
            :obj:`Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        config_dict = self.to_dict()

        # get the default config dict
        default_config_dict = PretrainedConfig().to_dict()

        # get class specific config dict
        class_config_dict = self.__class__().to_dict() if not self.is_composition else {}

        serializable_config_dict = {}

        # only serialize values that differ from the default config
        for key, value in config_dict.items():
            if (
                key not in default_config_dict
                or key == "transformers_version"
                or value != default_config_dict[key]
                or (key in class_config_dict and value != class_config_dict[key])
            ):
                serializable_config_dict[key] = value

        return serializable_config_dict

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            :obj:`Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type

        # # Transformers version when serializing the model
        # output["transformers_version"] = __version__

        return output

    def to_json_string(self, use_diff: bool = True) -> str:
        """
        Serializes this instance to a JSON string.

        Args:
            use_diff (:obj:`bool`, `optional`, defaults to :obj:`True`):
                If set to ``True``, only the difference between the config instance and the default
                ``PretrainedConfig()`` is serialized to JSON string.

        Returns:
            :obj:`str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike], use_diff: bool = True):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (:obj:`str` or :obj:`os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
            use_diff (:obj:`bool`, `optional`, defaults to :obj:`True`):
                If set to ``True``, only the difference between the config instance and the default
                ``PretrainedConfig()`` is serialized to JSON file.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string(use_diff=use_diff))

    def update(self, config_dict: Dict[str, Any]):
        """
        Updates attributes of this class with attributes from ``config_dict``.

        Args:
            config_dict (:obj:`Dict[str, Any]`): Dictionary of attributes that should be updated for this class.
        """
        for key, value in config_dict.items():
            setattr(self, key, value)

    def update_from_string(self, update_str: str):
        """
        Updates attributes of this class with attributes from ``update_str``.

        The expected format is ints, floats and strings as is, and for booleans use ``true`` or ``false``. For example:
        "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"

        The keys to change have to already exist in the config object.

        Args:
            update_str (:obj:`str`): String with attributes that should be updated for this class.

        """

        d = dict(x.split("=") for x in update_str.split(","))
        for k, v in d.items():
            if not hasattr(self, k):
                raise ValueError(f"key {k} isn't in the original config dict")

            old_v = getattr(self, k)
            if isinstance(old_v, bool):
                if v.lower() in ["true", "1", "y", "yes"]:
                    v = True
                elif v.lower() in ["false", "0", "n", "no"]:
                    v = False
                else:
                    raise ValueError(f"can't derive true or false from {v} (key {k})")
            elif isinstance(old_v, int):
                v = int(v)
            elif isinstance(old_v, float):
                v = float(v)
            elif not isinstance(old_v, str):
                raise ValueError(
                    f"You can only update int, float, bool or string values in the config, got {v} for key {k}"
                )

            setattr(self, k, v)

class GPT2Config(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a :class:`~transformers.GPT2Model` or a
    :class:`~transformers.TFGPT2Model`. It is used to instantiate a GPT-2 model according to the specified arguments,
    defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration
    to that of the GPT-2 `small <https://huggingface.co/gpt2>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 50257):
            Vocabulary size of the GPT-2 model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.GPT2Model` or
            :class:`~transformers.TFGPT2Model`.
        n_positions (:obj:`int`, `optional`, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        n_ctx (:obj:`int`, `optional`, defaults to 1024):
            Dimensionality of the causal mask (usually same as n_positions).
        n_embd (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the embeddings and hidden states.
        n_layer (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        n_head (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        n_inner (:obj:`int`, `optional`, defaults to None):
            Dimensionality of the inner feed-forward layers. :obj:`None` will set it to 4 times n_embd
        activation_function (:obj:`str`, `optional`, defaults to :obj:`"gelu"`):
            Activation function, to be selected in the list :obj:`["relu", "silu", "gelu", "tanh", "gelu_new"]`.
        resid_pdrop (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (:obj:`int`, `optional`, defaults to 0.1):
            The dropout ratio for the embeddings.
        attn_pdrop (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention.
        layer_norm_epsilon (:obj:`float`, `optional`, defaults to 1e-5):
            The epsilon to use in the layer normalization layers
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        summary_type (:obj:`string`, `optional`, defaults to :obj:`"cls_index"`):
            Argument used when doing sequence summary, used in the models :class:`~transformers.GPT2DoubleHeadsModel`
            and :class:`~transformers.TFGPT2DoubleHeadsModel`.

            Has to be one of the following options:

                - :obj:`"last"`: Take the last token hidden state (like XLNet).
                - :obj:`"first"`: Take the first token hidden state (like BERT).
                - :obj:`"mean"`: Take the mean of all tokens hidden states.
                - :obj:`"cls_index"`: Supply a Tensor of classification token position (like GPT/GPT-2).
                - :obj:`"attn"`: Not implemented now, use multi-head attention.
        summary_use_proj (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Argument used when doing sequence summary, used in the models :class:`~transformers.GPT2DoubleHeadsModel`
            and :class:`~transformers.TFGPT2DoubleHeadsModel`.

            Whether or not to add a projection after the vector extraction.
        summary_activation (:obj:`str`, `optional`):
            Argument used when doing sequence summary. Used in for the multiple choice head in
            :class:`~transformers.GPT2DoubleHeadsModel`.

            Pass :obj:`"tanh"` for a tanh activation to the output, any other value will result in no activation.
        summary_proj_to_labels (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Argument used when doing sequence summary, used in the models :class:`~transformers.GPT2DoubleHeadsModel`
            and :class:`~transformers.TFGPT2DoubleHeadsModel`.

            Whether the projection outputs should have :obj:`config.num_labels` or :obj:`config.hidden_size` classes.
        summary_first_dropout (:obj:`float`, `optional`, defaults to 0.1):
            Argument used when doing sequence summary, used in the models :class:`~transformers.GPT2DoubleHeadsModel`
            and :class:`~transformers.TFGPT2DoubleHeadsModel`.

            The dropout ratio to be used after the projection and activation.
        scale_attn_weights (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Scale attention weights by dividing by sqrt(hidden_size).
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return the last key/values attentions (not used by all models).

    Example::

        >>> from transformers import GPT2Model, GPT2Config

        >>> # Initializing a GPT2 configuration
        >>> configuration = GPT2Config()

        >>> # Initializing a model from the configuration
        >>> model = GPT2Model(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """

    model_type = "gpt2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=50257,
        n_positions=1024,
        n_ctx=1024,
        n_embd=768,
        n_cond=768,
        n_layer=12,
        n_head=12,
        n_inner=None,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        summary_type="cls_index",
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
        scale_attn_weights=True,
        gradient_checkpointing=False,
        use_cache=False,
        cond_vocab_size=None,
        cond_type='feed_forward',
        is_causal=False,
        use_fast_attn=False,
        bos_token_id=50256,
        eos_token_id=50256,
        **kwargs
    ):
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.n_ctx = n_ctx
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_cond = n_cond
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.summary_type = summary_type
        self.summary_use_proj = summary_use_proj
        self.summary_activation = summary_activation
        self.summary_first_dropout = summary_first_dropout
        self.summary_proj_to_labels = summary_proj_to_labels
        self.gradient_checkpointing = gradient_checkpointing
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.cond_vocab_size = cond_vocab_size
        self.cond_type = cond_type
        self.is_causal = is_causal
        self.use_fast_attn = use_fast_attn

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    @property
    def max_position_embeddings(self):
        return self.n_positions

    @property
    def hidden_size(self):
        return self.n_embd

    @property
    def num_attention_heads(self):
        return self.n_head

    @property
    def num_hidden_layers(self):
        return self.n_layer
