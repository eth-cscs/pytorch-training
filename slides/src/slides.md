---
addons:
  - slidev-addon-python-runner
theme: ./slidev-theme-cscs
# You can also start simply with 'default'
#theme: seriph
# random image from a curated Unsplash collection by Anthony
# like them? see https://unsplash.com/collections/94734566/slidev
background: https://cover.sli.dev
# some information about your slides (markdown enabled)
title: Hands-on Introduction to Deep Learning with PyTorch
info: |
  ## CSCS Course: Hands-on Introduction to Deep Learning with PyTorch
  Presentation slides for the CSCS course on Deep Learning with PyTorch.

  Sources available at [GitHub (eth-cscs/pytorch-training)](https://github.com/eth-cscs/pytorch-training)
favicon: /images/cscs.ico
# apply unocss classes to the current slide
#class: text-center
# https://sli.dev/features/drawing
drawings:
  persist: false
# slide transition: https://sli.dev/guide/animations.html#slide-transitions
transition: slide-left
# enable MDC Syntax: https://sli.dev/features/mdc
mdc: true
# open graph
# seoMeta:
#  ogImage: https://cover.sli.dev
# python runner config
python:
  installs: []
#fonts:
#  sans: Arial
#  seriph: Arial
#  mono: Victor Mono
lineNumbers: true
contextMenu: false
---

# Hands-on Introduction to Deep Learning with PyTorch

Swiss National Supercomputing Centre (CSCS) - ETH Zurich<br/>
2-4 July 2025, Lugano<br/>

<!--<div @click="$slidev.nav.next" class="mt-12 py-1" hover:bg="white op-10">-->
<!--  Press Space for next page <carbon:arrow-right />-->
<!--</div>-->

---
src: ./1.1-introduction/section-slides.md
hide: false
---

---
src: ./2.1-featurisation/section-slides.md
hide: false
---

---
src: ./2.2-cnn/section-slides.md
hide: false
---

---
src: ./3.1-more_on_training/section-slides.md
hide: false
---

transition: fade-out
---

