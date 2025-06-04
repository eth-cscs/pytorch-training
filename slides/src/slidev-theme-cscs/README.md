# slidev-theme-cscs

[![NPM version](https://img.shields.io/npm/v/slidev-theme-cscs?color=3AB9D4&label=)](https://www.npmjs.com/package/slidev-theme-cscs)

A CSCS theme for [Slidev](https://github.com/slidevjs/slidev).

<!--
  Learn more about how to write a theme:
  https://sli.dev/guide/write-theme.html
--->

<!-- Run `npm run dev` to check out the slides for more details of how to start writing a theme. -->

See it in action at https://eth-cscs.github.io/swe4py/slides/

## Install (current)

Submodule or copy the top-level folder into your slides project and add the following frontmatter to your `slides.md`.

<pre><code>---
theme: <b>./slidev-theme-cscs</b>
---</code></pre>

It has to be a *relative* path, but it can be elsewhere. The name of the directory can also be changed. If you are not planning to host the slides online you can also point to a single clone of this repo from multiple slidev projects.

## Install (once released on npm)

Add the following frontmatter to your `slides.md`. Start Slidev then it will prompt you to install the theme automatically.

<pre><code>---
theme: <b>cscs</b>
---</code></pre>

Learn more about [how to use a theme](https://sli.dev/guide/theme-addon#use-theme).

## Layouts

The builtin layouts should be styled as expected: https://sli.dev/builtin/layouts.

This theme provides two additional alternative cover layouts:

- cover-machine (same as the default cover layout)
- cover-building
- cover-formulae

## Components

This theme provides the following components:

```<TitleTop />``` will insert the logo header as seen in cover/end and section layouts.

## Contributing

- `npm install`
- `npm run dev` to start theme preview of `example.md`
- Edit the `example.md` and style to see the changes
- `npm run export` to generate the preview PDF
- `npm run screenshot` to generate the preview PNG
