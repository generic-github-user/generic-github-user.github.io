local utils = require("pandoc.utils")

local function ensure_identifier(el)
  if el.identifier and el.identifier ~= "" then
    return
  end

  local text = utils.stringify(el.content)
  local normalized = utils.normalize_identifier(text)
  if normalized == "" then
    normalized = "section-" .. pandoc.sha1(text)
  end
  el.identifier = normalized
end

function Header(el)
  if el.level ~= 2 then
    return nil
  end

  ensure_identifier(el)

  local anchor = pandoc.Link(
    {pandoc.Str("¶")},
    "#" .. el.identifier,
    "",
    {["class"] = "heading-anchor", ["title"] = "Link to this section"}
  )

  el.content:insert(#el.content + 1, pandoc.Space())
  el.content:insert(#el.content + 1, anchor)
  return el
end
