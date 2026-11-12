"""Offline Minecraft PDF rule self-test report; tutorials never enter the mean."""
import html
import json
from pathlib import Path


def escape(value):
    return html.escape(str(value), quote=True)


def render(root):
    root = Path(root)
    data = json.loads((root / 'result.json').read_text(encoding='utf-8'))
    summary = data['summary']
    mean = summary['mean_score']
    heading = f'完整计分例子平均分：{mean}' if mean is not None else '暂无可汇总的完整计分结果'
    rows = []
    details = []
    for index, episode in enumerate(data['episodes'], 1):
        scene = episode['scenario']
        metrics = episode.get('metrics', {})
        score = episode.get('scoring')
        total = score['score'] if score else '教学练习，不计分'
        rows.append('<tr>' + ''.join(f'<td>{escape(v)}</td>' for v in
                    (scene['label'], scene['mode'], episode['status'], episode['termination'],
                     metrics.get('action_count', '—'), total)) + '</tr>')
        directory = root / 'evidence' / str(index)
        links = ' · '.join(f'<a href="{escape(p.relative_to(root).as_posix())}">{escape(p.name)}</a>'
                           for p in sorted(directory.iterdir()) if p.is_file())
        video = directory / 'replay.mp4'
        player = f'<video controls preload="metadata" src="{escape(video.relative_to(root).as_posix())}"></video>' if video.exists() else ''
        table = ''
        if score:
            items = ''.join(f'<tr><td>{escape(k)}</td><td>{v if v is not None else "∞（未达成）"}</td></tr>'
                            for k, v in score['first_acquired'].items())
            table = ('<p>' + ' / '.join(f'{escape(k)}：{escape(score[k])}' for k in
                     ('tech', 'side', 'efficiency', 'death_penalty', 'score')) + '</p>'
                     '<table><tr><th>事项</th><th>首次达成步数</th></tr>' + items + '</table>'
                     f'<p>检查点：{escape(json.dumps(score["checkpoints"], ensure_ascii=False))}</p>')
        details.append(f'<section><h2>{escape(scene["label"])}</h2>{table}{player}<p>{links}</p></section>')
    document = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>无界探索 · 本地自测</title>'
                '<meta name="viewport" content="width=device-width"><style>'
                'body{max-width:1100px;margin:40px auto;padding:0 24px;font:16px/1.7 system-ui;color:#24382e;background:#f5f3ec}'
                'table{width:100%;border-collapse:collapse}td,th{padding:10px;text-align:left;border-bottom:1px solid #ccd3c8}'
                'section{margin:32px 0}video{width:100%}a{color:#405e45}</style>'
                f'<h1>无界探索 · 本地自测</h1><p>{escape(summary["notice"])}</p><h2>{escape(heading)}</h2>'
                '<p>前两个例子仅供教学。JSON 保留原始数值，不进行评分舍入。null 表示尚未达成（∞）。</p>'
                '<table><tr><th>例子</th><th>用途</th><th>状态</th><th>结束原因</th><th>完成步数</th><th>单局分数</th></tr>'
                + ''.join(rows) + '</table>' + ''.join(details) + '</html>')
    (root / 'review.html').write_text(document, encoding='utf-8')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('run_directory', type=Path)
    render(parser.parse_args().run_directory)
