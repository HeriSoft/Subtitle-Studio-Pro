import pysubs2, os, platform
def ensure_ass_from_srt_with_style(srt_path: str, ass_path: str, font_size: int=36, font_family: str="Arial",
                        color_hex: str="#FFFFFF", outline: float=2.0, shadow: float=0.5, pos: str="bottom", y_offset: int=40,
                        bold: bool=False, italic: bool=False, underline: bool=False, special_prefix: str='', special_suffix: str=''):
    subs = pysubs2.load(srt_path, encoding='utf-8')
    def hex_to_bgr(h):
        h = h.lstrip("#")
        r = int(h[0:2],16); g = int(h[2:4],16); b = int(h[4:6],16)
        return pysubs2.Color(b, g, r, 0)
    align = 2 if pos=="bottom" else 8
    st = pysubs2.SSAStyle()
    st.fontname = font_family
    st.fontsize = font_size
    st.primarycolor = hex_to_bgr(color_hex)
    st.outline = outline
    st.shadow = shadow
    st.marginl = st.marginr = 30
    st.marginv = max(0, y_offset)
    st.bold = 1 if bold else 0
    st.italic = 1 if italic else 0
    st.underline = 1 if underline else 0
    st.alignment = align
    subs.styles["Default"] = st
    if special_prefix or special_suffix:
        for ev in subs:
            ev.text = f"{special_prefix}{ev.text}{special_suffix}"
    subs.save(ass_path)
