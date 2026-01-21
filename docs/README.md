# Reagentic Documentation
<generated>
Welcome to the Reagentic framework documentation. This directory contains comprehensive guides and references for using the Reagentic AI agent framework.

## 📚 Documentation Index

### Core Concepts

- **[Subsystem Pattern](SUBSYSTEM_PATTERN.md)** - Comprehensive guide to the modular subsystem architecture
  - Learn how to create reusable, self-contained components
  - Understand tool categorization and agent integration
  - Explore advanced patterns and best practices

### Framework Features

- **[Logging System](LOGGING.md)** - Complete logging and observability guide
  - Structured logging with JSON format
  - Context tracking and performance monitoring
  - Integration with agents and providers

- **[Protocol System](PROTOCOL.md)** - Structured agent run logging and analysis
  - Event tracking (agent, LLM, tool, trace, span)
  - Automatic duration calculation
  - Multiple storage backends (SQLite, JSONLines)
  - Configurable detail levels

## 🚀 Quick Navigation

### Getting Started
- [Subsystem Pattern Basics](SUBSYSTEM_PATTERN.md#quick-start) - Create your first subsystem
- [Memory Subsystem Example](SUBSYSTEM_PATTERN.md#memory-subsystem-example) - Working with built-in memory
- [Protocol System Basics](PROTOCOL.md#quick-start) - Set up agent run logging

### Advanced Topics
- [Multi-Category Tools](SUBSYSTEM_PATTERN.md#multi-category-tools) - Organize tools by functionality
- [Event-Driven Subsystems](SUBSYSTEM_PATTERN.md#event-driven-subsystems) - Handle subsystem events
- [Integration Patterns](SUBSYSTEM_PATTERN.md#integration-patterns) - Combine multiple subsystems
- [Duration Tracking](PROTOCOL.md#duration-tracking) - Automatic timing for agent operations
- [Storage Backends](PROTOCOL.md#storage-backends) - SQLite and JSONLines options

### Best Practices
- [Architecture Guidelines](SUBSYSTEM_PATTERN.md#architecture) - Design patterns and principles
- [Error Handling](SUBSYSTEM_PATTERN.md#best-practices) - Robust error management
- [Troubleshooting](SUBSYSTEM_PATTERN.md#troubleshooting) - Common issues and solutions
- [Protocol Best Practices](PROTOCOL.md#best-practices) - Effective protocol logging

## 🏗️ Framework Architecture

The Reagentic framework is built around several key concepts:

1. **Subsystems** - Modular, reusable components that provide specific functionality
2. **Tools** - Agent capabilities exposed through the subsystem pattern
3. **Categories** - Logical groupings of tools for flexible agent configuration
4. **Events** - Reactive patterns for subsystem interaction and monitoring
5. **Protocol** - Structured logging of agent runs for debugging and analysis

## 📖 Documentation Standards

All documentation follows these standards:
- **Clear Examples**: Every concept includes working code examples
- **Type Safety**: Full typing information for all APIs
- **Best Practices**: Guidance on proper usage patterns
- **Troubleshooting**: Common issues and their solutions

## 🤝 Contributing

When adding new documentation:
1. Follow the established format and style
2. Include practical examples
3. Update this index with links to new documents
4. Ensure all code examples are tested and working

---

**Framework Version**: Latest  
**Last Updated**: Documentation reflects the current implementation 