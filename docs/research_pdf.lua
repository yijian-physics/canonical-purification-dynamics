-- Make linked code paths breakable in tables and keep report graphics in order.
function Code(el)
  local s = el.text:gsub("([\\{}%%#&$])", "\\%1"):gsub("_", "\\_{}"):gsub("/", "/\\allowbreak{}")
  return pandoc.RawInline("latex", "\\texttt{" .. s .. "}")
end
function Image(el)
  el.attributes.width = "100%"
  return el
end

-- The definitions table needs room for equations and a visible text gutter.
function Table(el)
  local header = ""
  for _, row in ipairs(el.head.rows) do
    for _, cell in ipairs(row.cells) do header = header .. pandoc.utils.stringify(cell.contents) .. " " end
  end
  if header:find("Input and dynamics", 1, true) then
    el.colspecs = {{pandoc.AlignLeft, 0.22}, {pandoc.AlignLeft, 0.41}, {pandoc.AlignLeft, 0.37}}
    return {
      pandoc.RawBlock("latex", "\\begingroup\\setlength{\\tabcolsep}{8pt}"),
      el,
      pandoc.RawBlock("latex", "\\endgroup")
    }
  end
end
