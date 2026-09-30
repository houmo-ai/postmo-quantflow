# vllm_merak

## 简介

本仓库实现一个vllm的后端插件， 具备如下功能：

1. 不走transformers的解析和load模型逻辑，而是实现一套能够解析自定义的模型config，并load模型的流程

## 开发约束

1. 严禁修改vllm库中的任何代码。
2. 仅允许在vllm_merak后端插件范围内修改代码。
