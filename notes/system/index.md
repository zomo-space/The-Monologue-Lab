---
layout: default
title: 系统技术
permalink: /system/
---

# 系统技术

笔记列表：

<ul>
  {%- assign pages_list = site.pages | where_exp: "p", "p.path contains 'system/'" | sort: "title" -%}
  {%- for p in pages_list -%}
    {%- if p.url != '/system/' and p.url != '/system/index.html' -%}
      <li><a href="{{ p.url | relative_url }}">{{ p.title | default: p.basename }}</a></li>
    {%- endif -%}
  {%- endfor -%}

  {%- assign static_list = site.static_files | where_exp: "f", "f.relative_path contains '/system/'" | sort: "name" -%}
  {%- for f in static_list -%}
    <li><a href="{{ f.relative_path | relative_url }}">{{ f.name }}</a></li>
  {%- endfor -%}
</ul>