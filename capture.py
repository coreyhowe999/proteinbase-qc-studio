"""Record screenshots + demo video of QC Studio (chatbox-builder layout). App live on :8521."""
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path("data/shots"); OUT.mkdir(parents=True, exist_ok=True)
VID = Path("data/video"); VID.mkdir(parents=True, exist_ok=True)


def ready(page, t=90):
    page.wait_for_selector(".stApp", timeout=t * 1000)
    for _ in range(t):
        if "QC Builder" in page.content():
            break
        time.sleep(2)
    page.wait_for_load_state("networkidle")
    time.sleep(4)


def aggrid(page):
    for fr in page.frames:
        try:
            if fr.locator(".ag-selection-checkbox").count() > 0:
                return fr
        except Exception:
            pass
    return None


def shot(page, f):
    page.screenshot(path=str(OUT / f)); print("saved", f)


def main():
    with sync_playwright() as p:
        br = p.chromium.launch(channel="chrome", headless=True)
        ctx = br.new_context(viewport={"width": 1680, "height": 1050}, device_scale_factor=1,
                             record_video_dir=str(VID),
                             record_video_size={"width": 1680, "height": 1050})
        page = ctx.new_page()
        page.goto("http://localhost:8521", wait_until="networkidle")
        ready(page)
        shot(page, "90_review.png")

        # select reference samples via checkboxes -> builder count rises
        gf = aggrid(page)
        if gf:
            for i in (2, 4, 6):
                try:
                    gf.locator(".ag-selection-checkbox").nth(i).click(); time.sleep(0.7)
                except Exception:
                    pass
        time.sleep(1.5)
        shot(page, "91_selected.png")

        # submit -> Claude finds the common pattern
        try:
            page.get_by_role("button", name="Submit").click()
            page.wait_for_selector("text=Pattern found", timeout=70000)
            time.sleep(2)
            page.mouse.wheel(0, 350); time.sleep(1.5)
            shot(page, "92_pattern.png")
            # save & apply as a filter across all ProteinBase
            page.get_by_role("button", name="Save & apply as filter").click()
            time.sleep(6)
            page.mouse.wheel(0, -600); time.sleep(1.5)
            shot(page, "93_filtered.png")
        except Exception as e:
            print("builder flow:", e)

        # QC Flags editable tab
        try:
            page.get_by_role("tab", name="QC Flags").click(); time.sleep(3)
            shot(page, "94_flags.png")
        except Exception as e:
            print("flags:", e)

        time.sleep(1)
        ctx.close(); br.close()

    vids = sorted(VID.glob("*.webm"))
    if vids:
        (VID / "qc_studio_demo.webm").write_bytes(vids[-1].read_bytes())
        print("video:", VID / "qc_studio_demo.webm")


if __name__ == "__main__":
    main()
