/// <reference types="vite/client" />

declare module '*.vue' {
  import type { DefineComponent } from 'vue'
  const component: DefineComponent<{}, {}, any>
  export default component
}

declare module 'elkjs/lib/elk.bundled.js' {
  import ELK from 'elkjs'
  export default ELK
}
