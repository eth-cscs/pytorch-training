import { defineConfig } from 'unocss'


export default defineConfig({
  rules: [
    ['title-border', {'border-bottom-color': '#a60b16', 'border-bottom-style': 'solid', 'border-bottom-width': '4px'}],
    ['below-header', {'position': 'absolute', 'top': '4.5rem'}]
  ]
})

