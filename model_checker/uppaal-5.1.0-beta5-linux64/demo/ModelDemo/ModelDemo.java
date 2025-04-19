import com.uppaal.engine.*;
import com.uppaal.engine.connection.BundledConnection;
import com.uppaal.model.core2.*;
import com.uppaal.model.io2.Problem;
import com.uppaal.model.io2.XTAReaderParsing.ParseException;
import com.uppaal.model.system.SystemEdge;
import com.uppaal.model.system.SystemLocation;
import com.uppaal.model.system.UppaalSystem;
import com.uppaal.model.system.concrete.ConcreteState;
import com.uppaal.model.system.concrete.ConcreteTrace;
import com.uppaal.model.system.concrete.Limit;
import com.uppaal.model.system.symbolic.SymbolicState;
import com.uppaal.model.system.symbolic.SymbolicTrace;
import com.uppaal.model.system.symbolic.SymbolicTransition;
import java.awt.geom.Point2D;
import java.io.*;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.ExecutionException;
import javax.xml.stream.XMLStreamException;

/**
 * This test exercises public API people might be using to interact with models and engine.
 */

public class ModelDemo
{
    /**
     * Valid kinds of labels on locations.
     */
    public enum LKind { name, init, urgent, committed, invariant, exponentialrate, comments }
    ;
    /**
     * Valid kinds of labels on edges.
     */
    public enum EKind { select, guard, synchronisation, assignment, comments }
    ;
    /**
     * Sets a label on a location.
     * @param l the location on which the label is going to be attached
     * @param kind a kind of the label
     * @param value the label value (either boolean or String)
     * @param x the x coordinate of the label
     * @param y the y coordinate of the label
     */
    public static Property addLabel(Location l, LKind kind, Object value, int x, int y)
    {
        Property p = l.setProperty(kind.name(), value);
        return p.setXY(x, y);
    }
    /**
     * Adds a location to a template.
     * @param t the template
     * @param name a name for the new location
     * @param exprate an expression for an exponential rate
     * @param x the x coordinate of the location
     * @param y the y coordinate of the location
     * @return the new location instance
     */
    public static Location addLocation(Template t, String name, String exprate, int x, int y)
    {
        Location l = t.addLocation();
        l.setXY(x, y);
        x += 8;
        if (name != null && !name.isBlank()) {
            y -= 25;
            addLabel(l, LKind.name, name, x, y);
        }
        if (exprate != null && !exprate.isBlank()) {
            y -= 18;
            addLabel(l, LKind.exponentialrate, exprate, x, y);
        }
        return l;
    }
    /**
     * Sets a label on an edge.
     * @param e the edge
     * @param kind the kind of the label
     * @param value the content of the label
     * @param x the x coordinate of the label
     * @param y the y coordinate of the label
     */
    public static Property addLabel(Edge e, EKind kind, String value, int x, int y)
    {
        Property p = e.setProperty(kind.name(), value);
        return p.setXY(x, y);
    }
    /**
     * Adds an edge to the template
     * @param t the template where the edge belongs
     * @param source the source location
     * @param target the target location
     * @param guard guard expression
     * @param sync synchronization expression
     * @param update update expression
     * @return
     */
    public static Edge addEdge(Template t, Location source, Location target, String guard, String sync, String update)
        throws ModelException
    {
        Edge e = t.addEdge(source, target);
        int x = (source.getX() + target.getX()) / 2 + 8;
        int y = (source.getY() + target.getY()) / 2;
        if (update != null && !update.isBlank()) {
            addLabel(e, EKind.assignment, update, x, y);
            y -= 18;
        }
        if (sync != null && !sync.isBlank()) {
            addLabel(e, EKind.synchronisation, sync, x, y);
            y -= 18;
        }
        if (guard != null && !guard.isBlank()) {
            addLabel(e, EKind.guard, guard, x, y);
        }
        return e;
    }

    public static Document createSampleModel() throws ModelException
    {
        // create a new Uppaal model with default properties:
        Document doc = new Document(new DocumentPrototype());
        // add global variables:
        doc.setProperty("declaration", "int v;\n\nclock x,y,z;");
        // add a TA template:
        Template t = doc.addTemplate();
        t.setProperty("name", "Experiment");
        // the template has initial location:
        Location l0 = addLocation(t, "L0", "1", 0, 0);
        l0.setProperty("init", true);
        // add another location to the right:
        Location l1 = addLocation(t, "L1", null, 162, 0);
        addLabel(l1, LKind.invariant, "x<=10", l1.getX() + 8, l1.getY());
        // add another location below to the right:
        Location l2 = addLocation(t, "L2", null, 162, 162);
        addLabel(l2, LKind.invariant, "y<=20", l2.getX() + 8, l2.getY());
        // add another location below:
        Location l3 = addLocation(t, "L3", "1", 0, 162);
        // add another location below:
        Location le = addLocation(t, "Error", "1", 0, 324);
        // add another location below:
        Location lg = addLocation(t, "Goal", null, -162, 162);
        // create an edge L0->L1 with an update and comment:
        addEdge(t, l0, l1, "v<2", null, "v=1,\nx=0")
            .setProperty(EKind.comments.name(), "Enabled only when v=1")
            .setY(l0.getY() - 64);
        addEdge(t, l0, le, null, null, null)
            .addNail(lg.getX() - 36, l0.getY())
            .addNail(lg.getX() - 36, le.getY())
            .setProperty(EKind.guard.name(), "v==6")
            .setXY(l0.getX() - 90, l0.getY() - 18);
        // create some more edges:
        addEdge(t, l1, l2, "x>=5", null, "v=2,\ny=0");
        addEdge(t, l2, l3, "y>=10", null, "v=3,\nz=0");
        addEdge(t, l3, l0, "x>=100", null, "v=4");
        addEdge(t, l3, le, "v==7", null, "v=6"); // unreachable!
        addEdge(t, l3, lg, "v==4", null, "v=5");
        // add system declaration:
        doc.setProperty("system",
            "Exp1 = Experiment();\n"
                + "Exp2 = Experiment();\n\n"
                + "system Exp1, Exp2;");
        QueryList ql = doc.getQueryList();
        ql.add(new Query("E<> Exp1.Goal", "Can Exp1 reach Goal?"));
        ql.add(new Query("E<> Exp1.Error", "Can Exp1 reach Error?"));
        ql.add(new Query("Pr[<=1000](<> Exp1.Goal)", "What is the probability of Exp1 reaching Goal?"));
        ql.add(new Query("simulate [<=1000] { v, x, y }", "get simulation trajectories"));
        return doc;
    }

    public static Document loadModel(String location) throws IOException { return Document.load(location); }

    /**
     * Connects to Uppaal engine and returns a handle to it.
     * @return
     * @throws EngineException
     * @throws IOException
     */
    public static Engine connectToEngine() throws EngineException, IOException
    {
        String uppaalHome = System.getenv("UPPAAL_HOME"); // /path/uppaal-5.0.1-linux64/
        if (uppaalHome == null) {
            File home = new File(".");
            while (home.getParentFile() != null) {
                if (new File(home, "uppaal.jar").exists())
                    uppaalHome = home.getAbsolutePath();
                else
                    home = home.getParentFile();
            }
            throw new IOException("Set UPPAAL_HOME environment variable to Uppaal installation path");
        }
        Engine engine = new Engine();
        engine.addConnection(new BundledConnection(new File(uppaalHome)));
        engine.connect();
        // engine.setServerPath(path);
        // engine.setServerHost("localhost");
        // engine.setConnectionMode(EngineStub.BOTH);
        // engine.connect();
        return engine;
    }

    /**
     * Compiles the document into a system representation.
     * @param engine
     * @param doc
     * @return
     * @throws EngineException
     * @throws IOException
     * @throws InterruptedException
     * @throws ExecutionException
     */
    public static UppaalSystem compile(Engine engine, Document doc)
        throws EngineException, IOException, InterruptedException, ExecutionException
    {
        // compile the model into system:
        ArrayList<Problem> problems = new ArrayList<>();
        UppaalSystem sys = engine.getSystem(doc, problems).get();
        if (!problems.isEmpty()) {
            boolean fatal = false;
            System.out.println("There are problems with the document:");
            for (Problem p : problems) {
                System.out.println(p.toString());
                if (!"warning".equals(p.getType())) { // ignore warnings
                    fatal = true;
                }
            }
            if (fatal) {
                System.exit(1);
            }
        }
        return sys;
    }

    /**
     * Prints symbolic state.
     * @param ps the print-writer to use
     * @param sys the system associated with the state
     * @param s
     */
    public static PrintStream print(PrintStream ps, UppaalSystem sys, SymbolicState s)
    {
        ps.print("(");
        boolean first = true;
        for (SystemLocation l : s.getLocations()) {
            if (first)
                first = false;
            else
                ps.print(", ");
            ps.print(l.getName());
        }
        int val[] = s.getVariableValues();
        for (int i = 0; i < sys.getNoOfVariables(); ++i) {
            ps.print(", ");
            ps.print(sys.getVariableName(i) + "=" + val[i]);
        }
        List<String> constraints = new ArrayList<>();
        s.getPolyhedron().getAllConstraints(constraints);
        for (String cs : constraints) {
            ps.print(", ");
            ps.print(cs);
        }
        ps.println(")");
        return ps;
    }

    /**
     * Prints the concrete state.
     * @param ps the print-writer to use
     * @param sys the system associated with the state
     * @param s the concrete state
     */
    public static PrintStream print(PrintStream ps, UppaalSystem sys, ConcreteState s)
    {
        ps.print("(");
        boolean first = true;
        for (SystemLocation l : s.getLocations()) {
            if (first)
                first = false;
            else
                ps.print(", ");
            ps.print(l.getName());
        }
        Limit limit = s.getInvariant();
        if (!limit.isUnbounded()) {
            if (limit.isStrict())
                ps.print(", limit<");
            else
                ps.print(", limit≤");
            ps.print(limit.getDoubleValue());
        }
        int offset = 0;
        int[] vals = s.getVars();
        for (int i = 0; i < vals.length; i++) {
            if (offset + i > 0)
                ps.print(", ");
            ps.print(sys.getVariableName(i) + " = " + vals[i]);
        }
        offset += vals.length;

        double[] fpvals = s.getFPVars();
        for (int i = 0; i < fpvals.length; i++) {
            if (offset + i > 0)
                ps.print(", ");
            ps.print(sys.getVariableName(offset + i) + " = " + fpvals[i]);
        }
        offset += fpvals.length;

        fpvals = s.getClocks();
        for (int i = 0; i < fpvals.length; i++) {
            if (offset + i > 0)
                ps.print(", ");
            ps.print(sys.getClockName(i) + " = " + fpvals[i]);
        }
        ps.println(")");
        return ps;
    }

    /**
     * Prints the symbolic trace as a sequence of states and transitions.
     * @param ps the print-stream to use.
     * @param sys the system associated with the trace.
     * @param trace the trace to print.
     */
    public static PrintStream printTrace(PrintStream ps, UppaalSystem sys, SymbolicTrace trace)
    {
        if (trace == null) {
            ps.println("(null trace)");
            return ps;
        }
        Iterator<SymbolicTransition> it = trace.iterator();
        print(ps, sys, it.next().getTarget());
        while (it.hasNext()) {
            SymbolicTransition trans = it.next();
            if (trans.getSize() == 0) {
                // no edges, something special (like "deadlock" or initial state):
                ps.println(trans.getEdgeDescription());
            } else {
                // one or more edges involved, print them:
                boolean first = true;
                for (SystemEdge e : trans.getEdges()) {
                    if (first)
                        first = false;
                    else
                        ps.print(", ");
                    ps.print(e.getProcessName() + ": " + e.getEdge().getSource().getPropertyValue("name") + " \u2192 "
                        + e.getEdge().getTarget().getPropertyValue("name"));
                }
            }
            ps.println();
            print(ps, sys, trans.getTarget());
        }
        ps.println();
        return ps;
    }

    /**
     * Prints the concrete trace as a sequence of states and transitions.
     * @param ps the print-stream to use.
     * @param sys the system associated with the trace.
     * @param trace the trace to print.
     */
    public static PrintStream printTrace(PrintStream ps, UppaalSystem sys, ConcreteTrace trace)
    {
        if (trace == null) {
            ps.println("(null trace)");
            return ps;
        }
        Iterator<ConcreteTraceElement> it = trace.iterator();
        print(ps, sys, it.next().getTarget());
        while (it.hasNext()) {
            ConcreteTraceElement trans = it.next();
            if (trans.getSize() == 0) {
                // no edges: "deadlock", initial state or last delay):
                ps.println(trans.getTransitionDescription());
            } else {
                // one or more edges involved, print them:
                boolean first = true;
                for (SystemEdge e : trans.getEdges()) {
                    if (first)
                        first = false;
                    else
                        ps.print(", ");
                    ps.print(e.getProcessName() + ": " + e.getEdge().getSource().getPropertyValue("name") + " \u2192 "
                        + e.getEdge().getTarget().getPropertyValue("name"));
                }
            }
            ps.println();
            print(ps, sys, trans.getTarget());
        }
        ps.println();
        return ps;
    }

    /**
     * Prints statistical data collected from SMC query.
     * @param ps
     * @param data
     */
    static public PrintStream print(PrintStream ps, QueryData data)
    {
        for (DataSet2D plot : result.getPlots()) {
            ps.println("Plot \"" + plot.getTitle() + "\" showing \"" + plot.getYLabel() + "\" over \""
                + plot.getXLabel() + "\"");
            for (Data2D traj : plot) {
                ps.print("Trajectory " + traj.getTitle() + ":");
                for (Point2D.Double p : traj) ps.print(" (" + p.x + "," + p.y + ")");
                ps.println();
            }
        }
        return ps;
    }

    /**
     * Simulates the system behavior using symbolic representation of states.
     * @param engine the Uppaal engine to use.
     * @param sys the compiled system model.
     * @return symbolic trace
     * @throws EngineException
     * @throws IOException
     * @throws CannotEvaluateException
     * @throws InterruptedException
     * @throws ExecutionException
     */
    public static SymbolicTrace symbolicSimulation(Engine engine, UppaalSystem sys)
        throws EngineException, IOException, CannotEvaluateException, InterruptedException, ExecutionException
    {
        PrintStream ps = System.out;
        SymbolicTrace trace = new SymbolicTrace();
        // compute the initial state:
        SymbolicState state = engine.getInitialState(sys).get();
        // add the initial transition to the trace:
        trace.add(new SymbolicTransition(null, null, state));
        while (state != null) {
            print(ps, sys, state);
            // compute the successors (including "deadlock"):
            List<SymbolicTransition> trans = engine.getTransitions(sys, state).get();
            // select a random transition:
            int n = (int) Math.floor(Math.random() * trans.size());
            SymbolicTransition tr = trans.get(n);
            // check the number of edges involved:
            if (tr.getSize() == 0) {
                // no edges, something special (like "deadlock"):
                ps.print(tr.getEdgeDescription());
            } else {
                // one or more edges involved, print them:
                boolean first = true;
                for (SystemEdge e : tr.getEdges()) {
                    if (first)
                        first = false;
                    else
                        ps.print(", ");
                    ps.print(e.getProcessName() + ": " + e.getEdge().getSource().getPropertyValue("name") + " \u2192 "
                        + e.getEdge().getTarget().getPropertyValue("name"));
                }
            }
            ps.println();
            // jump to a successor state (null in case of deadlock):
            state = tr.getTarget();
            // if successfull, add the transition to the trace:
            if (state != null)
                trace.add(tr);
        }
        return trace;
    }

    /**
     * Saves the symbolic trace in XTR format.
     * @param writer the stream-writer to use.
     * @param trace the symbolic trace
     * @throws IOException
     */
    public static OutputStreamWriter saveXTR(OutputStreamWriter writer, SymbolicTrace trace) throws IOException
    {
        /* BNF for the XTR format just in case
           (it may change, thus don't rely on it)
           <XTRFomat>  := <state> ( <state> <transition> ".\n" )* ".\n"
           <state>     := <locations> ".\n" <polyhedron> ".\n" <variables> ".\n"
           <locations> := ( <locationId> "\n" )*
           <polyhedron> := ( <constraint> ".\n" )*
           <constraint> := <clockId> "\n" clockId "\n" bound "\n"
           <variables> := ( <varValue> "\n" )*
           <transition> := ( <processId> <edgeId> )* ".\n"
        */
        Iterator<SymbolicTransition> it = trace.iterator();
        it.next().getTarget().writeXTRFormat(writer);
        while (it.hasNext()) {
            it.next().writeXTRFormat(writer);
        }
        writer.write(".\n");
        writer.flush();
        return writer;
    }

    static SymbolicTrace strace = null;
    static ConcreteTrace ctrace = null;

    public static void workflows(PrintStream ps, Document doc)
        throws EngineException, IOException, ExecutionException, InterruptedException, CannotEvaluateException
    {
        // connect to the engine server:
        Engine engine = connectToEngine();

        // compile the document into system representation:
        UppaalSystem sys = compile(engine, doc);

        // perform a random symbolic simulation and get a trace:
        ps.println("===== Random symbolic simulation =====");
        strace = symbolicSimulation(engine, sys);
        // save the trace to an XTR file:
        ps.println("===== Trace saving and loading =====");
        saveXTR(new FileWriter("ModelDemo.xtr"), strace).close();
        Parser parser = new Parser(new FileInputStream("ModelDemo.xtr"));
        // parse a trace from an XTR file:
        strace = parser.parseXTRTrace(sys);
        printTrace(ps, sys, strace);

        // Engine settings (see `verifyta --help`):
        EngineOptions options = engine.getOptions().get();
        EngineSettings settings = options.getDefaultSettings();
        ps.println("==== Engine Settings ===");
        settings.addSetting("--diagnostic", "0");
        settings.getProperties().forEach((k, v) -> ps.println("  " + k + ": " + v.getValue()));
        engine.setOptionSettings(settings);

        // simple model-checking:
        Query symb1 = new Query("E<> Exp1.Goal");
        ps.println("=== Symbolic check: " + symb1.getFormula() + " ===");
        strace = null;
        QueryResult res = engine.query(sys, symb1, qf).get();
        ps.println("Result: " + res); // OK
        printTrace(ps, sys, strace);

        Query symb2 = new Query("E<> Exp1.Error");
        ps.println("=== Symbolic check: " + symb2.getFormula() + " ===");
        strace = null;
        res = engine.query(sys, symb2, qf).get();
        ps.println("Result: " + res); // NOT_OK
        printTrace(ps, sys, strace);

        // Statistical model-checking:
        Query smc1 = new Query("Pr[<=1000](<> Exp1.Goal)");
        ps.println("=== SMC check: " + smc1.getFormula() + " ===");
        ctrace = null;
        res = engine.query(sys, smc1, qf).get();
        ps.println("Result: " + res);
        printTrace(ps, sys, ctrace);
        print(ps, res);

        Query smc2 = new Query("simulate [<=30] { v, x, y }");
        ps.println("=== Simulation: " + smc2.getFormula() + " ===");
        res = engine.query(sys, smc2, qf).get();
        ps.println("Result: " + res);
        print(ps, res);

        // Model-checking with customized initial state:
        // Find variable "v" in the system representation:
        int v_index = -1; // the index of variable v
        for (int i = 0; i < sys.getNoOfVariables(); ++i) {
            if ("v".equals(sys.getVariableName(i))) {
                v_index = i;
                break;
            }
        }

        // Customize the initial state:
        SymbolicState state = engine.getInitialState(sys).get();
        int[] vals = state.getVariableValues();
        if (v_index < 0 || v_index >= vals.length) {
            throw new IOException("Value for variable \"v\" was not found");
        }
        // set variable v to value 6:
        state.getVariableValues()[v_index] = 6;
        // add constrain "x-0<=5":
        state.getPolyhedron().addNonStrictConstraint(1, 0, 5);
        // add constrain "0-x<=-5" (equivalent to "x>=5"):
        state.getPolyhedron().addNonStrictConstraint(0, 1, -5);
        // Note that the constraints will propagate to other clocks
        // because of other (initial) constrains like: x==y, y==z

        ps.println("=== Custom state check: " + symb1.getFormula() + " === (expect NOT_OK)");
        strace = null;
        res = engine.query(sys, state, symb1, qf).get();
        ps.println("Result: " + res);
        printTrace(ps, sys, strace);

        ps.println("=== Custom state check: " + symb2.getFormula() + " === (expect OK)");
        strace = null;
        res = engine.query(sys, state, symb2, qf).get();
        ps.println("Result: " + res);
        printTrace(ps, sys, strace);
        engine.forceDisconnect(10).get();
    }

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args)
    {
        PrintStream ps = System.out;
        if (args.length < 1) {
            ps.println("This is a demo of Uppaal model.jar API");
            ps.println("Use one of the following arguments:");
            ps.println("  hardcoded");
            ps.println("  <URL>");
            ps.println("  <path/file.xml>");
            return;
        }
        try {
            Document doc;
            if ("hardcoded".equals(args[0])) {
                // create a hardcoded model:
                doc = createSampleModel();
            } else {
                // load model from a file/internet:
                doc = loadModel(args[0]);
            }
            // save the model into a file:
            doc.save("ModelDemo.xml");
            // load a model from the file:
            doc = loadModel("ModelDemo.xml");
            workflows(ps, doc);
            ps.println("Done!");
            System.exit(0);
        } catch (CannotEvaluateException | EngineException | IOException | InterruptedException | ExecutionException
            | ModelException ex) {
            ps.flush();
            ex.printStackTrace(System.err);
            System.exit(1);
        }
    }

    public static QueryFeedback qf = new QueryFeedback() {
        @Override
        public void setProgressAvail(boolean availability)
        {}

        @Override
        public void setProgress(int load, long vm, long rss, long cached, long avail, long swap, long swapfree,
            long user, long sys, long timestamp)
        {}

        @Override
        public void setSystemInfo(long vmsize, long physsize, long swapsize)
        {}

        @Override
        public void setLength(int length)
        {}

        @Override
        public void setCurrent(int pos)
        {}

        @Override
        public void setTrace(char result, String feedback, SymbolicTrace trace, QueryResult queryVerificationResult)
        {
            strace = trace;
        }

        public void setTrace(char result, String feedback, ConcreteTrace trace, QueryResult queryVerificationResult)
        {
            ctrace = trace;
        }
        @Override
        public void setFeedback(String feedback)
        {
            if (feedback != null && feedback.length() > 0) {
                System.out.println("Feedback: " + feedback);
            }
        }

        @Override
        public void appendText(String s)
        {
            if (s != null && s.length() > 0) {
                System.out.println("Append: " + s);
            }
        }

        @Override
        public void setResultText(String s)
        {
            if (s != null && s.length() > 0) {
                System.out.println("Result: " + s);
            }
        }
    };

    public static String toXMLString(Document doc) throws IOException
    {
        ByteArrayOutputStream xmlout = new ByteArrayOutputStream();
        doc.saveXML(xmlout);
        xmlout.close();
        return xmlout.toString();
    }

    public static Document fromXML(String xml) throws IOException, XMLStreamException
    {
        ByteArrayInputStream xmlin = new ByteArrayInputStream(xml.getBytes());
        return new DocumentPrototype().loadXML(xmlin, new ArrayList<Problem>());
    }

    public static String toXTAString(Document doc) throws IOException
    {
        ByteArrayOutputStream xtaout = new ByteArrayOutputStream();
        doc.saveXTA(xtaout, null);
        xtaout.close();
        return xtaout.toString();
    }

    public static Document fromXTA(String xta) throws IOException, XMLStreamException, ParseException
    {
        ByteArrayInputStream xtain = new ByteArrayInputStream(xta.getBytes());
        return new DocumentPrototype().loadXTA(xtain, null);
    }
}
