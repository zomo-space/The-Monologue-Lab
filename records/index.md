---
layout: default
title: 零零碎碎
permalink: /records/
---

# 零零碎碎

<a href="{{ '/README.html' | relative_url }}">README</a>

<ul>
  {%- assign pages_list = site.pages | where_exp: "p", "p.path contains 'records/'" | sort: "title" -%}
  {%- for p in pages_list -%}
    {%- if p.url != '/records/' and p.url != '/records/index.html' -%}
      <li><a href="{{ p.url | relative_url }}">{{ p.title | default: p.basename }}</a></li>
    {%- endif -%}
  {%- endfor -%}

  {%- assign static_list = site.static_files | where_exp: "f", "f.relative_path contains '/records/'" | sort: "name" -%}
  {%- for f in static_list -%}
    <li><a href="{{ f.relative_path | relative_url }}">{{ f.name }}</a></li>
  {%- endfor -%}
</ul>