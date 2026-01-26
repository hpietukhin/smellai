Apache EChart

An element to create a chart using ECharts. Updates can be pushed to the chart by changing the options property.
options:	dictionary of EChart options
on_point_click:	callback that is invoked when a point is clicked
on_click:	callback that is invoked when any component is clicked (added in version 3.5.0)
enable_3d:	enforce importing the echarts-gl library
renderer:	renderer to use ("canvas" or "svg", added in version 2.7.0)
theme:	an EChart theme configuration (dictionary or a URL returning a JSON object, added in version 2.15.0)
main.py

from nicegui import ui
from random import random

echart = ui.echart({
    'xAxis': {'type': 'value'},
    'yAxis': {'type': 'category', 'data': ['A', 'B'], 'inverse': True},
    'legend': {'textStyle': {'color': 'gray'}},
    'series': [
        {'type': 'bar', 'name': 'Alpha', 'data': [0.1, 0.2]},
        {'type': 'bar', 'name': 'Beta', 'data': [0.3, 0.4]},
    ],
})

def update():
    echart.options['series'][0]['data'][0] = random()

ui.button('Update', on_click=update)

ui.run()

NiceGUI
EChart with clickable points

You can register a callback for an event when a series point is clicked.
main.py

from nicegui import ui

ui.echart({
    'xAxis': {'type': 'category'},
    'yAxis': {'type': 'value'},
    'series': [{'type': 'line', 'data': [20, 10, 30, 50, 40, 30]}],
}, on_point_click=ui.notify)

ui.run()

NiceGUI
EChart with clickable components

Besides series points, you can register a callback for an event when any component registered with triggerEvent=True is clicked.

Hint: Check if that component is a point by checking e.component_type == 'series' to avoid double-processing with on_point_click.

Added in version 3.5.0
main.py

from nicegui import ui

ui.echart({
    'legend': {
        'triggerEvent': True,
    },
    'radar': {
        'triggerEvent': True,
        'indicator': [{'name': name, 'max': 100} for name in ['A', 'B', 'C']],
    },
    'series': [{
        'type': 'radar',
        'data': [{'name': 'Test', 'value': [77.0, 50.0, 90.0]}],
    }],
}, on_click=ui.notify)

ui.run()

NiceGUI
EChart with dynamic properties

Dynamic properties can be passed to chart elements to customize them such as apply an axis label format. To make a property dynamic, prefix a colon ":" to the property name.
main.py

from nicegui import ui

ui.echart({
    'xAxis': {'type': 'category'},
    'yAxis': {'axisLabel': {':formatter': 'value => "$" + value'}},
    'series': [{'type': 'line', 'data': [5, 8, 13, 21, 34, 55]}],
})

ui.run()

NiceGUI
EChart with custom theme

You can apply custom themes created with the Theme Builder.

Instead of passing the theme as a dictionary, you can pass a URL to a JSON file. This allows the browser to cache the theme and load it faster when the same theme is used multiple times.

Added in version 2.15.0
main.py

from nicegui import ui

ui.echart({
    'xAxis': {'type': 'category'},
    'yAxis': {'type': 'value'},
    'series': [{'type': 'bar', 'data': [20, 10, 30, 50, 40, 30]}],
}, theme={
    'color': ['#b687ac', '#28738a', '#a78f8f'],
    'backgroundColor': 'rgba(254,248,239,1)',
})

ui.run()

NiceGUI
EChart from pyecharts

You can create an EChart element from a pyecharts object using the from_pyecharts method. For defining dynamic options like a formatter function, you can use the JsCode class from pyecharts.commons.utils. Alternatively, you can use a colon ":" to prefix the property name to indicate that the value is a JavaScript expression.
main.py

from nicegui import ui
from pyecharts.charts import Bar
from pyecharts.commons.utils import JsCode
from pyecharts.options import AxisOpts

ui.echart.from_pyecharts(
    Bar()
    .add_xaxis(['A', 'B', 'C'])
    .add_yaxis('ratio', [1, 2, 4])
    .set_global_opts(
        xaxis_opts=AxisOpts(axislabel_opts={
            ':formatter': r'(val, idx) => `group ${val}`',
        }),
        yaxis_opts=AxisOpts(axislabel_opts={
            'formatter': JsCode(r'(val, idx) => `${val}%`'),
        }),
    )
)

ui.run()

NiceGUI
Run methods

You can run methods of the EChart instance using the run_chart_method method. This demo shows how to show and hide the loading animation, how to get the current width of the chart, and how to add tooltips with a custom formatter.

The colon ":" in front of the method name "setOption" indicates that the argument is a JavaScript expression that is evaluated on the client before it is passed to the method.
main.py

from nicegui import ui

echart = ui.echart({
    'xAxis': {'type': 'category', 'data': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']},
    'yAxis': {'type': 'value'},
    'series': [{'type': 'line', 'data': [150, 230, 224, 218, 135]}],
})

ui.button('Show Loading', on_click=lambda: echart.run_chart_method('showLoading'))
ui.button('Hide Loading', on_click=lambda: echart.run_chart_method('hideLoading'))

async def get_width():
    width = await echart.run_chart_method('getWidth')
    ui.notify(f'Width: {width}')
ui.button('Get Width', on_click=get_width)

ui.button('Set Tooltip', on_click=lambda: echart.run_chart_method(
    ':setOption', r'{tooltip: {formatter: params => "$" + params.value}}',
))

ui.run()

NiceGUI
Arbitrary chart events

You can register arbitrary event listeners for the chart using the on method and a "chart:" prefix. This demo shows how to register a callback for the "selectchanged" event which is triggered when the user selects a point.
main.py

from nicegui import ui

ui.echart({
    'toolbox': {'feature': {'brush': {'type': ['rect']}}},
    'brush': {},
    'xAxis': {'type': 'category'},
    'yAxis': {'type': 'value'},
    'series': [{'type': 'line', 'data': [1, 2, 3]}],
}).on('chart:selectchanged', lambda e: label.set_text(
    f'Selected point {e.args["fromActionPayload"]["dataIndexInside"]}'
))
label = ui.label()

ui.run()

