<script setup lang="ts">
import { watch, ref } from 'vue'
import { VueFlow, useVueFlow, Position } from '@vue-flow/core'
import { Background } from '@vue-flow/background'
import { Controls } from '@vue-flow/controls'
import ELK from 'elkjs/lib/elk.bundled.js'
import { useProtocolStore } from '../stores/protocol'
import type { GraphNode, GraphEdge } from '../types/protocol'
import ProtocolNode from './ProtocolNode.vue'

const store = useProtocolStore()
const { fitView } = useVueFlow()

const elk = new ELK()

const nodes = ref<any[]>([])
const edges = ref<any[]>([])

// Convert graph data to Vue Flow format with ELK layout
async function layoutGraph() {
  if (!store.graphData) {
    nodes.value = []
    edges.value = []
    return
  }

  const graphNodes = store.graphData.nodes
  const graphEdges = store.graphData.edges

  // ELK graph structure
  const elkGraph = {
    id: 'root',
    layoutOptions: {
      'elk.algorithm': 'layered',
      'elk.direction': 'DOWN',
      'elk.spacing.nodeNode': '50',
      'elk.layered.spacing.nodeNodeBetweenLayers': '80',
      'elk.layered.spacing.edgeNodeBetweenLayers': '30',
      'elk.padding': '[top=50,left=50,bottom=50,right=50]',
    },
    children: graphNodes.map((node: GraphNode) => ({
      id: node.id,
      width: 180,
      height: 70,
    })),
    edges: graphEdges.map((edge: GraphEdge) => ({
      id: edge.id,
      sources: [edge.source],
      targets: [edge.target],
    })),
  }

  try {
    const layoutedGraph = await elk.layout(elkGraph)
    
    // Apply positions to nodes
    nodes.value = graphNodes.map((node: GraphNode) => {
      const elkNode = layoutedGraph.children?.find((n: any) => n.id === node.id)
      return {
        id: node.id,
        type: 'protocol',
        position: {
          x: elkNode?.x ?? 0,
          y: elkNode?.y ?? 0,
        },
        data: {
          ...node.data,
          selected: node.data.entry_id === store.selectedEntryId,
        },
        sourcePosition: Position.Bottom,
        targetPosition: Position.Top,
      }
    })

    // Convert edges
    edges.value = graphEdges.map((edge: GraphEdge) => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      animated: edge.animated,
      style: {
        stroke: 'var(--border-color)',
        strokeWidth: 2,
      },
    }))

    // Fit view after layout
    setTimeout(() => {
      fitView({ padding: 0.2, duration: 300 })
    }, 50)
  } catch (e) {
    console.error('ELK layout error:', e)
  }
}

// Watch for graph data changes
watch(() => store.graphData, layoutGraph, { immediate: true })

// Update selected state when selection changes
watch(() => store.selectedEntryId, (newId) => {
  nodes.value = nodes.value.map(node => ({
    ...node,
    data: {
      ...node.data,
      selected: node.data.entry_id === newId,
    },
  }))
})

function onNodeClick(event: { node: any }) {
  const entryId = event.node.data.entry_id
  if (entryId) {
    store.selectEntry(entryId)
  }
}
</script>

<template>
  <div class="agent-graph">
    <div v-if="!store.hasSession" class="agent-graph__empty">
      <div class="empty-icon">📊</div>
      <div class="empty-text">Select a session to view the execution graph</div>
    </div>
    
    <VueFlow
      v-else
      :nodes="nodes"
      :edges="edges"
      :default-viewport="{ zoom: 1, x: 0, y: 0 }"
      :min-zoom="0.2"
      :max-zoom="2"
      fit-view-on-init
      @node-click="onNodeClick"
    >
      <template #node-protocol="nodeProps">
        <ProtocolNode v-bind="nodeProps" />
      </template>
      
      <Background pattern-color="var(--border-subtle)" :gap="20" />
      <Controls position="bottom-right" />
    </VueFlow>
  </div>
</template>

<style scoped>
.agent-graph {
  height: 100%;
  background: var(--bg-secondary);
  position: relative;
}

.agent-graph__empty {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 16px;
  color: var(--text-muted);
}

.empty-icon {
  font-size: 48px;
  opacity: 0.5;
}

.empty-text {
  font-size: 14px;
}

:deep(.vue-flow__controls) {
  background: var(--bg-tertiary);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-md);
  box-shadow: var(--shadow-subtle);
}

:deep(.vue-flow__controls-button) {
  background: transparent;
  border: none;
  color: var(--text-secondary);
  width: 28px;
  height: 28px;
}

:deep(.vue-flow__controls-button:hover) {
  background: var(--bg-elevated);
  color: var(--text-primary);
}

:deep(.vue-flow__controls-button svg) {
  fill: currentColor;
}
</style>
