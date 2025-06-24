<script setup>
import { Icon } from '@iconify/vue'

const props = defineProps({
  color: { type: String, default: 'amber-light' },
  title: { type: String, default: 'Note' },
  icon: { type: String, default: 'mdi-information-variant-circle-outline' },
  width: { type: String, default: '100%' },
})

const colorscheme = {
  'sky-light': 'bg-sky-100 border-sky-400 text-sky-900',
  'amber-light': 'bg-amber-100 border-amber-400 text-amber-900',
  'red-light': 'bg-red-100 border-red-400 text-red-900',
  'emerald-light': 'bg-emerald-100 border-emerald-400 text-emerald-900',
  'purple-light': 'bg-purple-100 border-purple-400 text-purple-900',
}

const fallbackClass = 'bg-gray-100 border-gray-400 text-gray-900'
</script>

<template>
  <div
    class="rounded border-l-4 p-4 my-4"
    :class="colorscheme[props.color] || fallbackClass"
    :style="{ width }"
  >
    <div class="flex items-center mb-2 font-semibold">
      <Icon :icon="props.icon" class="mr-2 text-lg" />
      {{ props.title }}
    </div>
    <div class="text-sm">
      <slot />
    </div>
  </div>
</template>

