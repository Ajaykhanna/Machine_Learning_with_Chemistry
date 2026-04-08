(function () {
  function tickStep(start, stop, count) {
    const step0 = Math.abs(stop - start) / Math.max(1, count);
    const power = Math.floor(Math.log10(step0 || 1));
    const error = step0 / Math.pow(10, power);
    const factor = error >= Math.sqrt(50) ? 10 : error >= Math.sqrt(10) ? 5 : error >= Math.sqrt(2) ? 2 : 1;
    return (stop < start ? -1 : 1) * factor * Math.pow(10, power);
  }

  function ticks(start, stop, count) {
    if (start === stop) {
      return [start];
    }
    const step = tickStep(start, stop, count);
    const values = [];
    const first = Math.ceil(start / step) * step;
    const last = Math.floor(stop / step) * step;
    for (let value = first; value <= last + step * 0.5; value += step) {
      values.push(Number(value.toFixed(12)));
    }
    if (!values.length) {
      values.push(start, stop);
    }
    return values;
  }

  function scaleLinear() {
    let domain = [0, 1];
    let range = [0, 1];

    function scale(value) {
      const [d0, d1] = domain;
      const [r0, r1] = range;
      if (d0 === d1) {
        return (r0 + r1) / 2;
      }
      const t = (value - d0) / (d1 - d0);
      return r0 + t * (r1 - r0);
    }

    scale.domain = function (values) {
      if (!arguments.length) return domain.slice();
      domain = values.slice();
      return scale;
    };

    scale.range = function (values) {
      if (!arguments.length) return range.slice();
      range = values.slice();
      return scale;
    };

    scale.invert = function (value) {
      const [d0, d1] = domain;
      const [r0, r1] = range;
      if (r0 === r1) {
        return (d0 + d1) / 2;
      }
      const t = (value - r0) / (r1 - r0);
      return d0 + t * (d1 - d0);
    };

    scale.ticks = function (count) {
      return ticks(domain[0], domain[1], count || 5);
    };

    return scale;
  }

  function extent(values) {
    let min = Infinity;
    let max = -Infinity;
    values.forEach((value) => {
      if (value < min) min = value;
      if (value > max) max = value;
    });
    return [min, max];
  }

  function max(values) {
    return extent(values)[1];
  }

  function min(values) {
    return extent(values)[0];
  }

  function pointer(event, node) {
    const rect = node.getBoundingClientRect();
    return [event.clientX - rect.left, event.clientY - rect.top];
  }

  window.d3 = {
    extent,
    max,
    min,
    pointer,
    scaleLinear,
    ticks
  };
})();
