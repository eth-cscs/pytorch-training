<!-- global-bottom.vue -->
<template>
  <LightOrDark>
    <template #light>
      <img 
        src="./assets/CSCS_logo_short.png"
        v-if="$nav.currentLayout !== 'cover' && $nav.currentLayout !== 'section' && $nav.currentLayout !== 'end'"
        class="absolute bottom-0 left-10 p-1 h-12 logo"
      />
      <img
        src="./assets/eth_logo_kurz_pos.png"
    v-if="$nav.currentLayout !== 'cover' && $nav.currentLayout !== 'section' && $nav.currentLayout !== 'end'"
        class="absolute bottom-0 right-10 p-1 h-12 logo"
      />
    </template>
    <template #dark>
      <img 
        src="./assets/CSCS_logo_short_neg.png"
        v-if="$nav.currentLayout !== 'cover' && $nav.currentLayout !== 'section' && $nav.currentLayout !== 'end'"
        class="absolute bottom-0 left-10 p-1 h-12 logo"
      />
      <img
        src="./assets/eth_logo_kurz_neg.png"
    v-if="$nav.currentLayout !== 'cover' && $nav.currentLayout !== 'section' && $nav.currentLayout !== 'end'"
        class="absolute bottom-0 right-10 p-1 h-12 logo"
      />
    </template>
  </LightOrDark>
  <footer
    class="absolute bottom-1 left-47% p-2"
    v-if="$nav.currentLayout !== 'cover' && $nav.currentLayout !== 'section' && $nav.currentLayout !== 'end'"
  >{{ $nav.currentPage }} / {{ $nav.total }}</footer>
</template>
