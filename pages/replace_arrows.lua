function Str(el)
  local replaced = el.text:gsub("->", "→")
  if replaced ~= el.text then
    el.text = replaced
    return el
  end
end
