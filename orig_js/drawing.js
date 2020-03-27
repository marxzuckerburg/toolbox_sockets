




function draw_net_v5(word,most_similar_data,width=600,height=600,io=undefined) {
	var d3 = require("d3@5")

	io.emit('status','formatting data...')
	var data = mostsim2netjson(most_similar_data,cutoff=DEFAULT_CSIM_CUTOFF)
	

	// 
	// chart = {
  const links = data.links.map(d => Object.create(d));
  const nodes = data.nodes.map(d => Object.create(d));

  const simulation = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id(d => d.id))
      .force("charge", d3.forceManyBody())
      .force("center", d3.forceCenter(width / 2, height / 2));

  const svg = d3.create("svg")
      .attr("viewBox", [0, 0, width, height]);

  const link = svg.append("g")
      .attr("stroke", "#999")
      .attr("stroke-opacity", 0.6)
    .selectAll("line")
    .data(links)
    .join("line")
      .attr("stroke-width", d => Math.sqrt(d.value));

  const node = svg.append("g")
      .attr("stroke", "#fff")
      .attr("stroke-width", 1.5)
    .selectAll("circle")
    .data(nodes)
    .join("circle")
      .attr("r", 5)
      .attr("fill", color)
      .call(drag(simulation));

  node.append("title")
      .text(d => d.id);

  simulation.on("tick", () => {
    link
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);

    node
        .attr("cx", d => d.x)
        .attr("cy", d => d.y);
  });

  invalidation.then(() => simulation.stop());

  return svg.node();
	// }

	// color = {
 // 	 const scale = d3.scaleOrdinal(d3.schemeCategory10);
 // 	 return d => scale(d.group);
	// }

	// drag = simulation => {
	//   function dragstarted(d) {
	//     if (!d3.event.active) simulation.alphaTarget(0.3).restart();
	//     d.fx = d.x;
	//     d.fy = d.y;
	//   }
	  
	//   function dragged(d) {
	//     d.fx = d3.event.x;
	//     d.fy = d3.event.y;
	//   }
	  
	//   function dragended(d) {
	//     if (!d3.event.active) simulation.alphaTarget(0);
	//     d.fx = null;
	//     d.fy = null;
	//   }
	  
	//   return d3.drag()
	//       .on("start", dragstarted)
	//       .on("drag", dragged)
	//       .on("end", dragended);
	// }


	// console.log('data',data);
	// console.log('chart',chart);
}









// function draw_net_d3n(word,most_similar_data,width=600,height=600,io=io,cutoff=DEFAULT_CSIM_CUTOFF) {
//  console.log('draw_net','word',word)
//  console.log('draw_net','most_similar_data',most_similar_data)
//  io.emit('status','drawing network...')

//  var d3 = require('d3');
//  // var jsdom = require('jsdom');
//  const D3Node = require('d3-node')

  


//  const d3n = new D3Node()      // initializes D3 with container element
//  svg = d3n.createSVG(width,height)

//  circle = svg.append('circle')
//        .attr('cx', 300)
//        .attr('cy', 150)
//        .attr('r', 30)
//        .attr('fill', '#26963c')

//  data = mostsim2netjson(word,most_similar_data)

//  // console.log('d3',d3)
//  // console.log('d3n',d3n)
//  // console.log('D3Node',D3Node)
//  var force = d3n.forceSimulation()
//                   .gravity(.05)
//                   .distance(100)
//                   .charge(-100)
//                   .size([width, height]);


//  return d3n.svgString() // output: <svg width=10 height=20 xmlns="http://www.w3.org/2000/svg"><g></g></svg>
// }








// drawing code
// const D3Node = require('d3-node')


// Pre-render d3 force-directed graph at server side
// Call node pre_render_d3_graph.js to generate d3_graph.html
// Original idea and framework borrowed from https://gist.github.com/mef/7044786



function draw_net(word,most_similar_data,width=600,height=600,io=io,cutoff=DEFAULT_CSIM_CUTOFF) {
  var d3 = require('d3')
  // var jsdom = require('jsdom')
  var fs = require('fs')
  var htmlStub = '<html><head> \
  <style>.node { stroke: #fff; fill: #ccc; stroke-width: 1.5px; } \
  .link { stroke: #333; stroke-opacity: .5; stroke-width: 1.5px; }</style> \
  </head><body><div id="dataviz-container"></div></body></html>'

  const jsdom = require("jsdom");
  const { JSDOM } = jsdom;

  const dom = new JSDOM(htmlStub); // `<!DOCTYPE html><p>Hello world</p>`);
  console.log(dom.window.document.querySelector("#dataviz-container")); // "Hello world"

  // const { window } = new JSDOM(htmlStub);
  // const { document } = (new JSDOM(htmlStub)).window;


    var data = mostsim2netjson(word,most_similar_data)
    var nodes = data['nodes']
    var links = data['links']

    console.log(nodes,links)

 
    var el = dom.window.document.querySelector('#dataviz-container')
    var body = dom.window.document.querySelector('body')

      console.log(el,body);
  
    // generate the graph
    // var width = 600,
    //     height = 600;

    width = dom.window.innerWidth;
    height = dom.window.innerHeight


    var svg = d3.select(el)
      .append('svg:svg')
      .attr('width', width)
      .attr('height', height);
    
    
     // circle = svg.append('circle')
     //       .attr('cx', 300)
     //       .attr('cy', 150)
     //       .attr('r', 30)
           // .attr('fill', '#26963c')

    
      const simulation = d3.forceSimulation()
      .force('charge', d3.forceManyBody().strength(-20)) 
      .force('center', d3.forceCenter(width / 2, height / 2))



      function getNodeColor(node) {
        return node.level === 1 ? 'red' : 'gray'
      }


    const nodeElements = svg.append('g')
      .selectAll('circle')
      .data(nodes)
      .enter().append('circle')
        .attr('r', 10)
        .attr('fill', '#26963c');
        // .attr('fill', getNodeColor)

        simulation.nodes(nodes).on('tick', function() {
        nodeElements
          .attr('cx', node => node.x)
          .attr('cy', node => node.y)
        // textElements
        //   .attr('x', node => node.x)
        //   .attr('y', node => node.y)
      });
  

      simulation.force('link', d3.forceLink()
        .id(link => link.id)
        .strength(link => link.strength))

      const linkElements = svg.append('g')
        .selectAll('line')
        .data(links)
        .enter().append('line')
          .attr('stroke-width', 1)
          .attr('stroke', '#E5E5E5')

                 linkElements
         .attr('x1', link => link.source.x)
         .attr('y1', link => link.source.y)
         .attr('x2', link => link.target.x)
         .attr('y2', link => link.target.y)


        // simulation.force('link').link(links)
      // // Here is the key. Without calling force.tick(), the simulation will not start and the nodes and links
      // will not have coordinates.
      for (var i = 0; i<1000; i++) 
        simulation.tick();
    
      return dom.window.document.querySelector('#dataviz-container').innerHTML;
    // } // end jsDom done callback
  // });

  // return svg.html();
}







function mostsim2netjson_V1(word,most_similar_data,cutoff=DEFAULT_CSIM_CUTOFF) {
  // format data
  var nodes = [{'id':0, 'label':word, 'group':1, 'level':1}];
  var node2index={}
  node2index[word]=0
  var links = [];

  most_similar_data.forEach(function(data) {
    console.log('data',data)
    word2=data['word'];
    csim=data['dist'];

    

    if (csim>=cutoff) {

      console.log(word,'--',csim,'-->',word2)

      if(!(word2 in node2index)) {
        index=nodes.length
        node2index[index]=word2
        
        new_node = {'id':index, 'label':word2, 'group':0, 'levels':0};
        nodes.push(new_node);

        links.push({'source':node2index[word], 'target':index, 'strength':csim})
      }

    }
  })

  console.log('nodes',nodes)
  console.log('links',links)

  //return (nodes,links);
  return {'nodes':nodes, 'links':links}
}





  // // Let's first initialize sigma:
  //   var s = new sigma('container');

  //   // Then, let's add some data to display:
  //   s.graph.addNode({
  //     // Main attributes:
  //     id: 'n0',
  //     label: 'Hello',
  //     // Display attributes:
  //     x: 0,
  //     y: 0,
  //     size: 1,
  //     color: '#f00'
  //   }).addNode({
  //     // Main attributes:
  //     id: 'n1',
  //     label: 'World !',
  //     // Display attributes:
  //     x: 1,
  //     y: 1,
  //     size: 1,
  //     color: '#00f'
  //   }).addEdge({
  //     id: 'e0',
  //     // Reference extremities:
  //     source: 'n0',
  //     target: 'n1'
  //   });

  //   // Finally, let's ask our sigma instance to refresh:
  //   s.refresh();

  //   console.log(s.graph,'!?')



