"""Native ribbon fit, real analyses, measurement persistence and report export."""
from contextlib import ExitStack
from pathlib import Path
import json
import os
import sys
import tempfile
import time
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('QT_QPA_PLATFORM','windows' if sys.platform=='win32' else 'xcb')
import numpy as np
import trimesh
from PySide6.QtCore import QSettings, Qt, QRect
from PySide6.QtGui import QFontMetrics
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QCheckBox, QToolButton, QFrame
import main_window
import app_updater
from project_store import ProjectState,save_project,load_project


def main():
    output=ROOT/('output/analysis-ribbon-smoke');output.mkdir(parents=True,exist_ok=True)
    app=QApplication([]);report={'status':'running','checks':[]}
    with tempfile.TemporaryDirectory() as folder,ExitStack() as patches:
        settings=QSettings(str(Path(folder)/'settings.ini'),QSettings.IniFormat);settings.setFallbacksEnabled(False)
        patches.enter_context(patch.object(main_window,'QSettings',lambda *args:settings))
        for method in ('start','check','manual_check'):patches.enter_context(patch.object(app_updater.UpdateController,method,lambda *args:None))
        window=main_window.MainWindow();window.resize(1600,1000);window.show()
        try:
            a=trimesh.creation.box([10,10,10]);a.apply_translation([0,0,5]);b=a.copy();b.apply_translation([6,0,0])
            window.restore_project(ProjectState(page='slicer',platforms=[dict(name='Build',dim=[100,100,100],is_default=True)],parts=[dict(mesh=a,filename='A.stl',platform='Build'),dict(mesh=b,filename='B.stl',platform='Build')]))
            window.ui.scene_tabs.setCurrentIndex(1);window.reset_history()
            for width in (1600,1300,1000,800):
                window.resize(width,950);QTest.qWait(80)
                assert window.width()==width,(width,window.width())
                tabs=window.ui.magics_ribbon
                for i in range(tabs.count()):
                    tabs.setCurrentIndex(i);QTest.qWait(30);page=tabs.widget(i);viewport=page.viewport()
                    window.grab(QRect(0,0,width,tabs.height()+45)).save(str(output/f'ribbon-{width}-{i}.png'))
                    separators=[frame for frame in page.findChildren(QFrame) if frame.frameShape()==QFrame.VLine or (frame.width()<=3 and frame.height()>30)]
                    for button in page.findChildren(QToolButton):
                        if button.toolButtonStyle()!=Qt.ToolButtonTextUnderIcon:continue
                        assert button.iconSize().width()==28 and button.iconSize().height()==28
                        metrics=QFontMetrics(button.font());lines=button.text().split('\n')
                        assert button.height()>=28+len(lines)*metrics.lineSpacing()+8,(tabs.tabText(i),button.text(),button.height())
                        point=button.mapTo(viewport,button.rect().topLeft())
                        assert point.y()>=0 and point.y()+button.height()<=viewport.height(),(tabs.tabText(i),button.text(),point.y(),viewport.height())
                        rect=QRect(point,button.size())
                        for separator in separators:
                            line_rect=QRect(separator.mapTo(viewport,separator.rect().topLeft()),separator.size())
                            assert not rect.adjusted(-6,0,6,0).intersects(line_rect),(width,tabs.tabText(i),button.text(),rect,line_rect)
                    bar=page.horizontalScrollBar()
                    if bar.maximum():
                        bar.setValue(bar.maximum());QTest.qWait(10)
                        window.grab(QRect(0,0,width,tabs.height()+45)).save(str(output/f'ribbon-{width}-{i}-right.png'))
                        bar.setValue(0)
            report['checks'].append('Every ribbon fits at widths 800/1000/1300/1600; separators never overlap buttons; icons are 28x28; captions are not clipped')
            if '--ribbon-only' in sys.argv:
                report['status']='passed';print('RIBBON_LAYOUT_SMOKE_OK');return
            tabs.setCurrentIndex(next(i for i in range(tabs.count()) if tabs.tabText(i)=='АНАЛИЗ И ОТЧЕТЫ'))
            tools=window.analysis_tools
            assert len(tools.buttons['view'].menu().actions())==7
            tools.buttons['view'].menu().actions()[1].trigger()
            def job():
                end=time.monotonic()+40
                while window._job is not None:
                    assert time.monotonic()<end,'analysis timeout';QTest.qWait(10)
                app.processEvents()
            def click(op):tools.buttons[op].click();app.processEvents()
            click('intersections');job();assert tools.results['intersections'][1]['result'][0]['volume']==400.
            with patch('analysis_tools.QInputDialog.getItem',return_value=('+Z',True)),patch('analysis_tools.QInputDialog.getDouble',return_value=(20.,True)):click('trapping');job()
            assert 'trapping' in tools.results
            with patch('analysis_tools.QInputDialog.getDouble',return_value=(11.,True)):click('walls');job()
            assert tools.results['walls'][1]['result'][0]['thin']>0
            assert any(name.startswith('analysis_walls_') for name in window.ui.slicer_plotter.actors)
            plotter=window.ui.slicer_plotter;source=plotter.actors[window.slicer_parts[0]['actor_name']]
            source.SetVisibility(False);tools.sync_highlights();assert not plotter.actors['analysis_walls_0'].GetVisibility()
            source.SetVisibility(True);tools.sync_highlights();assert plotter.actors['analysis_walls_0'].GetVisibility()
            original=window.slicer_parts[0]['mesh'];window.slicer_parts[0]['mesh']=original.copy()
            tools.sync_highlights();assert 'analysis_walls_0' not in plotter.actors
            window.slicer_parts[0]['mesh']=original
            tools.clear_highlights();assert not any(name.startswith('analysis_walls_') for name in window.ui.slicer_plotter.actors)
            click('cavities');job();assert tools.results['cavities'][1]['result'][0]['shells'][0]['closed']
            with patch('analysis_tools.QInputDialog.getDouble',return_value=(1.,True)):click('slices');job()
            assert tools.results['slices'][1]['result']['area_mm2'][0]==200.
            tools.dialog.grab().save(str(output/'slice-chart.png'))
            report['checks'].append('Real background intersections, directional extraction, wall thickness/highlights, shells and slice chart succeed')
            with patch.object(tools,'configuration',lambda:True):click('time');click('cost')
            assert 'Оценка времени' in tools.dialog.report.toPlainText()
            click('distance');assert window.ui.measurement_panel.active==(0,0)
            click('thickness');panel=window.ui.measurement_panel;assert panel.active==(0,4)
            face=int(np.flatnonzero(a.face_normals[:,0]>.9)[0]);panel.hits=[(0,face,a.triangles_center[face])];panel.calculate();panel.stop()
            assert '10.0000' in panel.results.item(0).text()
            with patch('analysis_tools.QInputDialog.getInt',return_value=(6,True)):click('precision')
            panel.active=(0,0);panel.hits=[(0,face,np.array([0.,0.,0.])),(0,face,np.array([1.234567,0.,0.]))];panel.calculate();panel.stop()
            assert '1.234567' in panel.results.item(1).text()
            with patch('analysis_tools.QInputDialog.getText',return_value=('Размер A',True)),patch('analysis_tools.QInputDialog.getItem',return_value=('мм',True)),patch('analysis_tools.QInputDialog.getDouble',side_effect=[(10.,True),(10.05,True),(.1,True)]):click('actual')
            assert tools.annotations[0]['passed'];window.travel_history(-1);assert tools.annotations==[];window.travel_history(1);assert len(tools.annotations)==1
            with patch('analysis_tools.QInputDialog.getText',side_effect=[('Отчёт проверки',True),('Оператор',True)]):click('template')
            click('report');assert 'Отчёт проверки' in tools.dialog.report.toPlainText()
            text=tools.dialog.report.toPlainText()
            from analysis_geometry import build_estimate
            data=dict(title=tools.title,estimate=build_estimate([dict(mesh=a,filename='A',supports=[])],tools.params))
            for kind in ('html','pdf','json','csv'):
                path=Path(folder)/('report.'+kind)
                with patch('analysis_tools.QFileDialog.getSaveFileName',return_value=(str(path),'')):tools.export(data,text,kind)
                assert path.is_file() and path.stat().st_size>50,kind
                if kind=='pdf':assert path.read_bytes().startswith(b'%PDF')
            saved=Path(folder)/'analysis.mrp';save_project(saved,window.capture_project());window.restore_project(load_project(saved))
            assert len(tools.annotations)==1 and panel.decimals==6 and tools.title=='Отчёт проверки'
            report['checks'].append('Distance, one-click thickness and 6-digit precision work; actual values, report settings, Undo/Redo and MRP roundtrip persist')
            report['checks'].append('HTML, PDF, JSON and CSV reports are written and PDF has valid header')
            report['status']='passed'
        finally:
            (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            if window.analysis_tools.dialog:window.analysis_tools.dialog.close()
            window.dirty=False;window.close();window.deleteLater();app.processEvents()
    print('ANALYSIS_RIBBON_SMOKE_OK')


if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        error=traceback.format_exc();(ROOT/'output/analysis-ribbon-smoke/error.txt').write_text(error,encoding='utf-8');print(error,file=sys.__stderr__);raise SystemExit(1)
