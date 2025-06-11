<template>
  <div class="slidev-layout cover">
    <TitleTop />
    <div class="my-auto w-full below-header">
      <LightOrDark>
        <template #light><img src="../assets/cover-machine.png" class="banner"/></template>
        <template #dark><img src="../assets/cover-building.png" class="banner"/></template>
      </LightOrDark>
      <slot />
    </div>
  </div>
</template>
