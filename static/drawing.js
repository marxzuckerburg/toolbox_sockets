
var colors = ["#1b9e77","#d95f02","#7570b3","#e7298a","#66a61e","#e6ab02","#a6761d","#666666",
              "#1b9e77","#d95f02","#7570b3","#e7298a","#66a61e","#e6ab02","#a6761d","#666666",
              "#1b9e77","#d95f02","#7570b3","#e7298a","#66a61e","#e6ab02","#a6761d","#666666",
              "#1b9e77","#d95f02","#7570b3","#e7298a","#66a61e","#e6ab02","#a6761d","#666666",
              "#1b9e77","#d95f02","#7570b3","#e7298a","#66a61e","#e6ab02","#a6761d","#666666",
              "#1b9e77","#d95f02","#7570b3","#e7298a","#66a61e","#e6ab02","#a6761d","#666666",
              "#1b9e77","#d95f02","#7570b3","#e7298a","#66a61e","#e6ab02","#a6761d","#666666",
              "#1b9e77","#d95f02","#7570b3","#e7298a","#66a61e","#e6ab02","#a6761d","#666666",
              "#1b9e77","#d95f02","#7570b3","#e7298a","#66a61e","#e6ab02","#a6761d","#666666",
              "#1b9e77","#d95f02","#7570b3","#e7298a","#66a61e","#e6ab02","#a6761d","#666666",
              "#1b9e77","#d95f02","#7570b3","#e7298a","#66a61e","#e6ab02","#a6761d","#666666",
              "#1b9e77","#d95f02","#7570b3","#e7298a","#66a61e","#e6ab02","#a6761d","#666666",
              "#1b9e77","#d95f02","#7570b3","#e7298a","#66a61e","#e6ab02","#a6761d","#666666",
              "#1b9e77","#d95f02","#7570b3","#e7298a","#66a61e","#e6ab02","#a6761d","#666666"]




function spaces_highlight_line(word) {
  // d3.select("svg").selectAll("*:not(.b_class)");
  d3v4.selectAll('.line').transition().style('opacity', 0.25);
  d3v4.selectAll('.dot').transition().style('opacity', 0.25);
  d3v4.selectAll('.word_label').transition().style('opacity', 0.25);
  d3v4.selectAll('.word_legend').transition().style('opacity', 0.25);
  // $('.word_label').style('opacity',0);


  d3v4.selectAll('.dot_' + word).transition().style('opacity', 1);
  d3v4.selectAll('.word_label_'+word).transition().style('opacity', 1);
  d3v4.selectAll('.word_legend_'+word).transition().style('opacity', 1);
  d3v4.selectAll('.line_'+word).transition().style('opacity', 1);
  // d3v4.selectAll('.line_' + word).transition().style('opacity', 1);
}

function spaces_unhighlight_lines() {
  d3v4.selectAll('.line').transition().style('opacity', 1);
  d3v4.selectAll('.dot').transition().style('opacity', 1);
  d3v4.selectAll('.word_label').transition().style('opacity', 1);
  d3v4.selectAll('.word_legend').transition().style('opacity', 1);

}


function zscore(array) {
  mean = math.mean(array)
  std = math.std(array)

  var newarr=[]
  array.forEach(function(x) { z = ((x-mean)/std); newarr.push(z) })

  return newarr
}




function plot_dynamic(data_ld,
  word_col='word',
  y_col = "umap_V2",
  x_col = "umap_V1",
  t_col = "period",

  div_id="umap",
  
  x_min=undefined,
  x_max=undefined,
  y_min=undefined,
  y_max=undefined,
  y_mid=undefined,
  x_mid=undefined,
  sp_title="",
  y_col_name = "",
  x_col_name = "",
  
  orig_width=666,
  orig_height=666) {



  // get data
  var WORDS = []
  var XVALS = []
  var YVALS = []
  var TVALS = []

  data_ld.forEach(function(d) { 
    WORDS.push(d[word_col])
    XVALS.push(d[x_col])
    YVALS.push(d[y_col])
    TVALS.push(d[t_col])
  })

  console.log('WORDS',WORDS)
  console.log('XVALS',XVALS)
  console.log('YVALS',YVALS)
  console.log('TVALS',TVALS)

  XVALS=zscore(XVALS)
  YVALS=zscore(YVALS)
  console.log('XVALS',XVALS)
  console.log('YVALS',YVALS)


  x_std=math.std(XVALS)
  y_std=math.std(YVALS)

  if(x_min==undefined) { x_min = math.min(XVALS) - (x_std/1.5) }
  if(x_max==undefined) { x_max = math.max(XVALS) + (x_std/1.5)}
  if(x_mid==undefined) { x_mid = math.median(XVALS) }
  if(y_min==undefined) { y_min = math.min(YVALS) - (y_std/1.5)}
  if(y_max==undefined) { y_max = math.max(YVALS) + (y_std/1.5)}
  if(y_mid==undefined) { y_mid = math.median(YVALS) }

  // x_min -= (x_min/1.5)
  // x_max += (x_min/1.5)
  // y_min -= (y_min/1.5)
  // y_max += (y_min/1.5)

  x_mid = x_min + ((x_max-x_min)/1.5) 
  y_mid = y_min + ((y_max-y_min)/1.5) 

  console.log('minmax',x_min,x_mid,x_max)
  console.log('minmax',y_min,y_mid,y_max)


  // SET UP FIGURE
  var margin = {top: 0, right: 0, bottom: 30, left: 0},
  width = orig_width - margin.left - margin.right,
  height = orig_height - margin.top - margin.bottom;
  
  //create figure
  $('#'+div_id).html("")
  var svg = d3v4.select('#'+div_id)
    .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform","translate(" + margin.left + "," + margin.top + ")")
  
  // create axes
  var x = d3v4.scaleLinear().domain([x_min, x_max]).range([ 0, width ]);
  var y = d3v4.scaleLinear().domain( [y_min, y_max]).range([ height, 0 ]);
  svg.append("g").attr("transform", "translate(0," + height + ")").call(d3v4.axisBottom(x)).call(d3v4.axisTop(x));
  svg.append("g").call(d3v4.axisLeft(y));
  svg.append("g").call(d3v4.axisRight(y));

  // text label for the x axis
  svg.append("text").attr("transform","translate(" + (width/1.5) + " ," +(height + margin.top + 20) + ")")
     .style("text-anchor", "middle")
     .text(function() { if(x_col_name != "") { return x_col_name; } else { return x_col; } })
     .attr('fill','black')
     .attr('class','x_axis_label axis_label')
     .on('click',function() { describe_field(x_col) });

  // text label for the y axis
  svg.append("text")
     .attr('class','y_axis_label axis_label')
     .attr("transform", "rotate(-90)")
     .attr("y", 0 - margin.left + 5)
     .attr("x",0 - (height / 2))
     .attr('fill','black')
     .attr("dy", "1em")
     .style("text-anchor", "middle")
     .text(function() { if(y_col_name != "") { return y_col_name; } else { return y_col; } })
     .on('click',function() { describe_field(y_col) });

  // create crosshairs
  svg.append("line").style("stroke", "black").attr("x1", 0).attr("y1", y(y_mid)).attr("x2", width).attr("y2", y(y_mid));
  svg.append("line").style("stroke", "black").attr("x1", x(x_mid)).attr("y1", 0).attr("x2", x(x_mid)).attr("y2", height);



  // compile data in graph format
  DATA = []
  data_byword = {}
  
  

  words_list = []
  WORDS.forEach(function(w,di) {
    dat={'date':parseFloat(XVALS[di]), 'value':parseFloat(YVALS[di]),'word':w, 
    // 'color':colorScale(di),
    'period':TVALS=[di] }
    if(!(w in data_byword)) { data_byword[w]=[]; words_list.push(w) }
    data_byword[w].push(dat)
    DATA.push(dat)
  })

  const colorScale = d3v4.scaleOrdinal().domain([0,data_byword.length]).range(
    // ["#A07A19", "#AC30C0", "#EB9A72", "#BA86F5", "#EA22A8"])
    ["#1b9e77","#d95f02","#7570b3","#e7298a","#66a61e","#e6ab02","#a6761d","#666666"])
  

  var word_i=-1
  for(word in data_byword) {
    word_i++
    data=data_byword[word]
    data_length = DATA.length;
  
    triangleScale = d3v4.scaleLinear().domain([0,data_length]).range([2,1]);

    for (ti=0; ti<data_length; ti++) {
      svg.append("svg:defs").append("svg:marker")
         .attr("id", 'triangle_'+word+'_'+(ti+1).toString())
         .attr("refX", 6)
         .attr("refY", 6)
         .attr("markerWidth", 30)
         .attr("markerHeight", 30)
         .attr("markerUnits","userSpaceOnUse")
         .attr("orient", "auto")
         .append("path")
         .attr("d", "M 0 0 12 6 0 12 3 6");
         // .style("fill", data[ti].color);
    }


    var lineScale = d3v4.scaleLinear().domain([0, 1]).range([10, 1]);
    var lineScaleCircle = d3v4.scaleLinear().domain([0, 1]).range([10, 3]);

    var all_periods = false
    for (di=0; di<=data.length; di++) {
        dat1=data[di];
        dat2=data[di+1];

        // Draw lines?
        if (dat1!= undefined & dat2!=undefined) {
          var all_periods=true
          //// console.log(dat1.date, dat1.value, dat2.date, dat2.value);
          svg.append('line')
             .style("stroke", colorScale(word_i))
             .style("stroke-width", lineScale((di+1)/data.length))
             .attr("x1", x(dat1.date))
             .attr("y1", y(dat1.value))
             .attr("x2", x(dat2.date))
             .attr("y2", y(dat2.value))
             .attr('class','line line_'+dat1.word);
      }
    }


    svg.append("g")
       .selectAll("dot")
       .data(data)
       .enter()
       .append("circle")
       .attr("cx", function(d) { return x(d.date) } )
       .attr("cy", function(d) { return y(d.value) } )
       .attr("r", 5) //function(d,i) { return lineScaleCircle((i+1)/data.length); })  //function(d,i) { return 3.5; } )
       // .attr("stroke",color)
       .attr('stroke-width',1)
       .attr("stroke",colorScale(word_i))
       // .attr("fill",colorScale(word_i))
       .attr("fill",'transparent')
       .attr('class','dot dot_'+word+' '+'dot')
       .on('mouseover', function (d, i) {
          desc_neighb(d.word,d.period);
          spaces_highlight_line(d.word);
        })
        .on('mouseout', function(d) { spaces_unhighlight_lines(); })
        .on('click', function(d,i) {
          get_ranks(word,clear=false,redirect=false,origin_id=false,popup=true);
        });


    // console.log(data,'?')
    svg.append("g")
       .selectAll("dot")
       .data(data)
       .enter()
       .append("text")
       .attr("x", function(d) { return x(d.date)+5; })
       .attr("y", function(d) { return y(d.value)-5; })
       .text(word)
       .attr('class','word_label word_label_'+word)
       .on('mouseover', function (d, i) {
          if (all_periods==false) { res=desc_neighb(word,'_total'); }
          spaces_highlight_line(word);
       })
       .on('mouseout', function(d,i) {
          spaces_unhighlight_lines();
       })
       .on('click', function(d,i) {
          get_ranks(word,clear=false,redirect=false,origin_id=false,popup=true);
       })
       // .attr('fill',function() {
          // if (all_periods!=false) { return 'black'; } else { return color; }
       // })
      .attr('style',function(d,i) { if (i==(d.length-1) || i==0) { return "" } else { return "display:none;"; }  })
      .attr("font-size", "11px");
  }


  svg.append("g")
  .selectAll("dot")
  .data(words_list)
  .enter()
  .append("text")
  .attr("x", x(x_max)-100)
  .attr("y", function(_d,_i) { return y(y_max) + (_i * 10) + 25 ; } )
  .attr('class',function (d, i) { return 'word_legend word_legend_'+word; })
  .text(function(_d,_i) { return _d; })
  .on('mouseover', function (word) {
    if (all_periods==false) { res=desc_neighb(word,'_total'); }
    spaces_highlight_line(word)
  })
  .on('mouseout', function() {
    spaces_unhighlight_lines();
  })
  .on('click', function(word) {
    get_ranks(word,clear=false,redirect=false,origin_id=false,popup=true);
  })
  .attr('fill',function(word,_i) { return colorScale(_i) })
  .attr("font-size", "13px")
  // .attr('style',function(d,i){ if (w_i!=0) { return "display: none;" } else { return ""; } })
  .attr("font-weight","bold");
}















      // // Add the points

      // function desc_neighb(_word,_period) {
      //   // $('')
      //   $.getJSON('/static/data/db/neighborhoods/COHA_30yr/'+_word+'.json', function(json) {
      //     //_period=_period.toString() + '-' + (parseInt(_period) + 30).toString()
      //     ////// console.log(json);
      //     ////// console.log(_period, json[_period] );

      //     // $('#progressbar').html("")
      //     // var bar = new ProgressBar.Circle(progressbar, {
      //     //   color: '#aaa',
      //     //   // This has to be the same size as the maximum width to
      //     //   // prevent clipping
      //     //   strokeWidth: 6,
      //     //   trailWidth: 1,
      //     //   // easing: 'easeInOut',
      //     //   duration: 0,
      //     //   text: {
      //     //     autoStyleContainer: false
      //     //   },
      //     //   from: { color: '#aaa', width: 1 },
      //     //   to: { color: '#333', width: 4 },
      //     //   // Set default step function for all animate calls
      //     //   step: function(state, circle) {
      //     //     circle.path.setAttribute('stroke', state.color);
      //     //     circle.path.setAttribute('stroke-width', state.width);

      //     //     var value = Math.round(circle.value() * 100);
      //     //     if (value === 0) {
      //     //       circle.setText('');
      //     //     } else {
      //     //       circle.setText(value+'%');
      //     //     }

      //     //   }
      //     // });
      //     // // bar.text.style.fontFamily = '"Raleway", Helvetica, sans-serif';
      //     // bar.text.style.fontSize = '1rem';
      //     // bar.animate(0.0);



      //     for (pwi=0; pwi<json.length; pwi++) {
      //       bar.animate((pwi+1)/json.length);
      //       period=json[pwi][0];
      //       pwords=json[pwi][1];

      //       pstr1 = pwords.slice(0,5).join(", ")
      //       pstr2 = pwords.slice(5,10).join(", ")
      //       pstr3 = pwords.slice(10,15).join(", ")

      //       if (period.startsWith(_period.toString())) {
      //         ////// console.log(period,_period,pwords);
      //         //return pwords;

      //         d3v4.selectAll('.neighb_window').remove();

      //         svg.append("text")
      //             .attr("transform",
      //             "translate(" + 25 + " ," +
      //                                  (height + margin.top - 25 ) + ")")
      //             .style("text-anchor", "left")
      //             .attr('class','neighb_window')
      //             .style('text-align', 'left')
      //             .attr('fill',color)
      //             .text(pstr3);

      //         svg.append("text")
      //             .attr("transform",
      //             "translate(" + 25 + " ," +
      //                                  (height + margin.top - 50 ) + ")")
      //             .style("text-anchor", "left")
      //             .attr('class','neighb_window')
      //             .style('text-align', 'left')
      //             .attr('fill',color)
      //             .text(pstr2);

      //             svg.append("text")
      //                 .attr("transform",
      //                 "translate(" + 25 + " ," +
      //                                      (height + margin.top - 75 ) + ")")
      //                 .style("text-anchor", "left")
      //                 .attr('class','neighb_window')
      //                 .style('text-align', 'left')
      //                 .attr('fill',color)
      //                 .text(pstr1);

      //             svg.append("text")
      //                 .attr("transform",
      //                       // "translate(" + ((width/1.5)) + " ," +
      //                       "translate(" + 25 + " ," +
      //                                      (height + margin.top - 100 ) + ")")
      //                 .style("text-anchor", "left")
      //                 .attr('class','neighb_window')
      //                 .text(_word +" ("+period+")")
      //                 .style('text-align', 'left')
      //                 .attr('fill',color)
      //                 .attr('style','font-weight: bold;');
      //       }
      //     }
      //   });
      // }









// (function() {
function make_slopegraph(word,silent=false) {
  $('#slopegraph').html( $('#slopegraph').html() + "\n<div class='slopediv'><div class=\"popup_closer\"> <a href=\"#\" onclick=\"close_popups()\">[Close]</a></div> <h2>Changing associations of “"+word+"”</h2>\n <li>The top 20 associated words per half-century.</li>\n<li>Click a word in the graph to pivot to that word's own data.</li><br/><div class=\"slope\" style=\"width: 1200px; height: 800px\" id='slopegraph_"+word+"'></div></div>");

$("#slopegraph_"+word).html("");

  if (silent==true) { $('#slopegraph').hide(); }

    'use strict';

    var data,
 keyValues = ['1810-1840', '1840-1870', '1870-1900', '1900-1930', '1930-1960', '1960-1990', '1990-2020'];

 //console.log(keyValues);

    // store chart
    var slopegraph;
    // track any user interactions
    var state = {
        // have an array to mutate
        keys: keyValues,
        // track filtered sets
        filter: [],
        // toggle highlights
        navToggle: [],
        // track line selection
        highlight: null
    };

    d3.json('/static/data/db/ranks/COHA_30yr/'+word+'.json', function(error, json) {
      //console.log('hi?',json);
        if (error) throw error;
        // access data outside this callback
        data = json;
        // initial render chart
        render(data, keyValues);
        // alternative navigation
        navAlt(data);
        // add some filter options
        filterFunc();
    });

    // filter sets via user interaction
    function filterFunc() {
        // create array values
        _.times(keyValues.length, function(n) {
            state.filter.push(true);
        });

        d3.select('#filter').append('ul')
            .selectAll('li')
            .data(keyValues)
            .enter().append('li')
            .on('click', function (d, i) {
                if (!state.filter[i]) {
                    // set toggle
                    state.filter[i] = true;
                    d3.select(this).style('opacity', 1);
                    // push key into array
                    state.keys.push(d);
                    // ensure array is kept in date order
                    state.keys = _.sortBy(state.keys);
                    // render chart with new keys
                    render(data, state.keys);
                // ensure there at least two values
                // so a slopegraph can be rendered
                } else if (state.filter[i] && state.keys.length > 2) {
                    state.filter[i] = false;
                    d3.select(this).style('opacity', 0.3);
                    _.pull(state.keys, d);
                    state.keys = _.sortBy(state.keys);
                    render(data, state.keys);
                }
            })
            .text(function (d) { return d; });
    }

    // navigation to highlight lines
    function navAlt(data) {
        // create array values
        _.times(data.length, function(n) {
            state.navToggle.push(true);
        });

        d3.select('#nav-alt').append('ul')
            .selectAll('li')
            .data(data)
            .enter().append('li')
            .attr('class', function (d, i) { return 'navAlt li-' + i; })
            .on('click', function (d, i) {
                if (!state.navToggle[i]) {
                    // update toggle state
                    state.navToggle[i] = true;
                    resetSelection();
                    state.highlight = null;
                } else if (state.navToggle[i]) {
                    state.navToggle[i] = false;
                    // hover to highlight line
                    highlightLine(i);
                    // highlight nav in relation to line
                    highlightNav(i);
                    // update state
                    state.highlight = i;
                }
            })
            .text(function (d) { return d['word']; });
    }


    // render slopegraph chart
    function render(data, keys) {
        resetSelection();
        // create chart
        slopegraph = d3.eesur.slopegraph_v2()
            .margin({top: 20, bottom: 20, left: 100, right: 100})
            .gutter(25)
            .keyName('word')
            .keyValues(keys)
            .on('_hover', function (d, i) {
                // hover to highlight line
                highlightLine(i);
                // highlight nav in relation to line
                highlightNav(i);
                // update state of selected highlight line
                state.highlight = i;
            });


        // apply chart
        d3.select('#slopegraph_'+word)
            .datum(data)
            .call(slopegraph);

        // ensure highlight is maintained on update
        if (!_.isNull(state.highlight)) {
            d3.selectAll('.elm').style('opacity', 0.33);
            d3.selectAll('.sel-' + state.highlight).style('opacity', 1);
            highlightNav(state.highlight);
        }
    }

    function highlightLine(i) {
        d3.selectAll('.elm').transition().style('opacity', 0.33);
        d3.selectAll('.sel-' + i).transition().style('opacity', 1);
        // d3.selectAll('.elm').transition().style('stroke', 'black');
        // d3.selectAll('.elm').transition().style('stroke-width', 1);
        // d3.selectAll('.sel-' + i).transition().style('stroke', 'red');
        // d3.selectAll('.sel-' + i).transition().style('stroke-width', 1);
    }

    function highlightNav(i) {
        d3.selectAll('.navAlt').transition().style('opacity', 0.6);
        d3.select('.li-' + i).transition().style('opacity', 1);
    }

    function resetSelection() {
        d3.selectAll('.elm').transition().style('opacity', 1);
        d3.selectAll('.navAlt').transition().style('opacity', 1);
    }

    // just for blocks viewer size
    d3.select(self.frameElement).style('height', '800px');


    //$('#analyze_word').val(word);
    // //console.log(word);
//
// if (silent == true) {
//
//  //console.log('silent...')
//
// //     $.magnificPopup.open({
// //       items: {
// //         src: $('<div class="popup-fullscreen" id="ranks-popup"></div>'),
// //         type: 'inline'
// //     },
// //
// //
// //     callbacks: {
// //     open: function() {
// //       //console.log('called back!', $('#ranks-popup').html());
// //
// //       $('#ranks-popup').html("");
// //       $('#slopegraph').appendTo('#ranks-popup');
// //
// //       //console.log('called back!', $('#ranks-popup').html(),  'slopegraph:', $('#slopegraph').html());
// //
// //       $('#slopegraph').show();
// //
// //       // $('#slopegraph').appendTo('.ranks-popup');
// //       // $('#slopegraph').show();
// //       // //console.log(item); // Do whatever you want with "item" object
// //     }
// //   }
// // });
//
//
// } else {
//  $('#slopegraph').show();
// }

}


// make_slopegraph();
