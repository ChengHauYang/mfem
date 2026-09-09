//                       MFEM Example 1 - Parallel Version
//
// Compile with: make ex1p
//
// Sample runs:  mpirun -np 4 ex1p -m ../data/square-disc.mesh
//               mpirun -np 4 ex1p -m ../data/star.mesh
//               mpirun -np 4 ex1p -m ../data/star-mixed.mesh
//               mpirun -np 4 ex1p -m ../data/escher.mesh
//               mpirun -np 4 ex1p -m ../data/fichera.mesh
//               mpirun -np 4 ex1p -m ../data/fichera-mixed.mesh
//               mpirun -np 4 ex1p -m ../data/toroid-wedge.mesh
//               mpirun -np 4 ex1p -m ../data/octahedron.mesh -o 1
//               mpirun -np 4 ex1p -m ../data/periodic-annulus-sector.msh
//               mpirun -np 4 ex1p -m ../data/periodic-torus-sector.msh
//               mpirun -np 4 ex1p -m ../data/square-disc-p2.vtk -o 2
//               mpirun -np 4 ex1p -m ../data/square-disc-p3.mesh -o 3
//               mpirun -np 4 ex1p -m ../data/square-disc-nurbs.mesh -o -1
//               mpirun -np 4 ex1p -m ../data/star-mixed-p2.mesh -o 2
//               mpirun -np 4 ex1p -m ../data/disc-nurbs.mesh -o -1
//               mpirun -np 4 ex1p -m ../data/pipe-nurbs.mesh -o -1
//               mpirun -np 4 ex1p -m ../data/ball-nurbs.mesh -o 2
//               mpirun -np 4 ex1p -m ../data/fichera-mixed-p2.mesh -o 2
//               mpirun -np 4 ex1p -m ../data/star-surf.mesh
//               mpirun -np 4 ex1p -m ../data/square-disc-surf.mesh
//               mpirun -np 4 ex1p -m ../data/inline-segment.mesh
//               mpirun -np 4 ex1p -m ../data/amr-quad.mesh
//               mpirun -np 4 ex1p -m ../data/amr-hex.mesh
//               mpirun -np 4 ex1p -m ../data/mobius-strip.mesh
//               mpirun -np 4 ex1p -m ../data/mobius-strip.mesh -o -1 -sc
//
// Device sample runs:
//               mpirun -np 4 ex1p -pa -d cuda
//               mpirun -np 4 ex1p -fa -d cuda
//               mpirun -np 4 ex1p -pa -d occa-cuda
//               mpirun -np 4 ex1p -pa -d raja-omp
//               mpirun -np 4 ex1p -pa -d ceed-cpu
//               mpirun -np 4 ex1p -pa -d ceed-cpu -o 4 -a
//               mpirun -np 4 ex1p -pa -d ceed-cpu -m ../data/square-mixed.mesh
//               mpirun -np 4 ex1p -pa -d ceed-cpu -m ../data/fichera-mixed.mesh
//             * mpirun -np 4 ex1p -pa -d ceed-cuda
//             * mpirun -np 4 ex1p -pa -d ceed-hip
//               mpirun -np 4 ex1p -pa -d ceed-cuda:/gpu/cuda/shared
//               mpirun -np 4 ex1p -pa -d ceed-cuda:/gpu/cuda/shared -m ../data/square-mixed.mesh
//               mpirun -np 4 ex1p -pa -d ceed-cuda:/gpu/cuda/shared -m ../data/fichera-mixed.mesh
//               mpirun -np 4 ex1p -pa -d ceed-cpu -m ../data/beam-tet.mesh
//
// Device simplices sample runs:
//               mpirun -np 4 ex1p -pa -d gpu -m ../data/inline-tet.mesh
//               mpirun -np 4 ex1p -pa -d gpu -m ../data/inline-tri.mesh
//
// Built-in Cartesian mesh and manufactured solution runs (-dim 2|3 replaces -m
// with the unit square / unit cube that the manufactured solutions live on):
//               mpirun -np 8 ex1p -dim 3 -n 16 -o 3 -l2 -mms sine -no-vis -no-pv -no-out
//               mpirun -np 8 ex1p -dim 3 -n 24 -o 4 -l2 -mms bubble-exp -pa -d ceed-cpu -a
//               mpirun -np 8 ex1p -dim 3 -n 64 -o 1 -l2 -mms multimode -no-pa
//               mpirun -np 8 ex1p -dim 2 -n 128 -o 6 -l2 -mms multimode
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Poisson problem
//               -Delta u = 1 with homogeneous Dirichlet boundary conditions.
//               Specifically, we discretize using a FE space of the specified
//               order, or if order < 1 using an isoparametric/isogeometric
//               space (i.e. quadratic for quadratic curvilinear mesh, NURBS for
//               NURBS mesh, etc.)
//
//               The example highlights the use of mesh refinement, finite
//               element grid functions, as well as linear and bilinear forms
//               corresponding to the left-hand side and right-hand side of the
//               discrete linear system. We also cover the explicit elimination
//               of essential boundary conditions, static condensation, and the
//               optional connection to the GLVis tool for visualization.

#include "mfem.hpp"
#include <cstring>
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

real_t sine_exact_solution(const Vector &x);
real_t sine_rhs(const Vector &x);
real_t multimode_exact_solution(const Vector &x);
real_t multimode_rhs(const Vector &x);
real_t bubble_exp_exact_solution(const Vector &x);
real_t bubble_exp_rhs(const Vector &x);

int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init();
   const double cold_start_begin = MPI_Wtime();
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Hypre::Init();

   // 2. Parse command-line options.
   // const char *mesh_file = "../data/star.mesh";
   const char *mesh_file = "../data/inline-quad.mesh";
   // int order = 1;
   int order = 6;
   int serial_ref_levels = 0;
   int mesh_dim = 0;
   int cells_per_direction = 16;
   bool static_cond = false;
   // bool pa = false;
   bool pa = true;
   bool fa = false;
   // const char *device_config = "cpu";
   const char *device_config = "ceed-cpu";
   bool visualization = true;
   bool paraview = true;
   bool save_output = true;
   bool l2_error = false;
   const char *mms = "sine";
   int profile_repeats = 0;
   double cg_relative_tolerance = 1e-12;
   int cg_max_iterations = 2000;
   // bool algebraic_ceed = false;
   bool algebraic_ceed = true;
   bool ceed_assembled_coarse_solver = false;

#ifdef MFEM_USE_CUDSS
   bool cudss_solver = false;
#endif

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&serial_ref_levels, "-rs", "--refine-serial",
                  "Number of serial uniform refinements.");
   args.AddOption(&mesh_dim, "-dim", "--dimension",
                  "Build a Cartesian mesh of the unit square (2, quads) or of"
                  " the unit cube (3, hexes) instead of reading --mesh;"
                  " 0 (default) reads the mesh file.");
   args.AddOption(&cells_per_direction, "-n", "--cells-per-direction",
                  "Cells per direction of the built-in Cartesian mesh.");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&fa, "-fa", "--full-assembly", "-no-fa",
                  "--no-full-assembly", "Enable Full Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
#ifdef MFEM_USE_CEED
   args.AddOption(&algebraic_ceed, "-a", "--algebraic",
                  "-no-a", "--no-algebraic",
                  "Use algebraic Ceed solver");
   args.AddOption(&ceed_assembled_coarse_solver,
                  "-ac", "--assembled-coarse-solver",
                  "-no-ac", "--no-assembled-coarse-solver",
                  "Use assembled BoomerAMG on the coarsest CEED AMG level.");
#endif
#ifdef MFEM_USE_CUDSS
   args.AddOption(&cudss_solver, "-cudss", "--cudss-solver", "-no-cudss",
                  "--no-cudss-solver", "Use the cuDSS Solver.");
#endif
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&paraview, "-pv", "--paraview", "-no-pv",
                  "--no-paraview", "Enable or disable ParaView output.");
   args.AddOption(&save_output, "-out", "--output", "-no-out",
                  "--no-output", "Enable or disable mesh and solution output.");
   args.AddOption(&l2_error, "-l2", "--l2-error", "-no-l2",
                  "--no-l2-error",
                  "Use a smooth manufactured solution on the unit square/cube"
                  " and compute its L2 error.");
   args.AddOption(&mms, "-mms", "--manufactured-solution",
                  "Manufactured solution: sine, multimode, or bubble-exp.");
   args.AddOption(&profile_repeats, "-pr", "--profile-repeats",
                  "Number of additional same-process solves to profile.");
   args.AddOption(&cg_relative_tolerance, "-rtol", "--relative-tolerance",
                  "Relative tolerance for CG.");
   args.AddOption(&cg_max_iterations, "-max-it", "--max-iterations",
                  "Maximum number of CG iterations.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (strcmp(mms, "sine") != 0 && strcmp(mms, "multimode") != 0 &&
       strcmp(mms, "bubble-exp") != 0)
   {
      if (myid == 0)
      {
         cerr << "Unknown manufactured solution: " << mms << endl;
      }
      return 1;
   }
   if (cg_relative_tolerance <= 0.0 || cg_max_iterations <= 0)
   {
      if (myid == 0)
      {
         cerr << "CG relative tolerance and maximum iterations must be positive."
              << endl;
      }
      return 1;
   }
   if (mesh_dim != 0 && mesh_dim != 2 && mesh_dim != 3)
   {
      if (myid == 0)
      {
         cerr << "Dimension must be 0 (read --mesh), 2 or 3." << endl;
      }
      return 1;
   }
   if (mesh_dim != 0 && cells_per_direction <= 0)
   {
      if (myid == 0)
      {
         cerr << "Cells per direction must be positive." << endl;
      }
      return 1;
   }
   if (profile_repeats < 0)
   {
      if (myid == 0)
      {
         cerr << "Profile repeats must be non-negative." << endl;
      }
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   if (myid == 0)
   {
      device.Print();
   }
   const double cold_start_seconds = MPI_Wtime() - cold_start_begin;

   // 4. Build the (serial) mesh on all processors. With -dim 2 or -dim 3 we
   //    generate a Cartesian mesh of the unit square or the unit cube, which is
   //    the domain the manufactured solutions below are defined on; -dim 3 is
   //    the cheapest way to reach large DOF counts. Otherwise we read the mesh
   //    from the given mesh file: we can handle triangular, quadrilateral,
   //    tetrahedral, hexahedral, surface and volume meshes with the same code.
   Mesh mesh = [&]() -> Mesh
   {
      const int n = cells_per_direction;
      if (mesh_dim == 3)
      {
         return Mesh::MakeCartesian3D(n, n, n, Element::HEXAHEDRON);
      }
      if (mesh_dim == 2)
      {
         return Mesh::MakeCartesian2D(n, n, Element::QUADRILATERAL);
      }
      return Mesh(mesh_file, 1, 1);
   }();
   int dim = mesh.Dimension();

   // 5. Refine the serial mesh on all processors to increase the resolution.
   //    The number of uniform refinements is controlled by -rs.
   for (int l = 0; l < serial_ref_levels; l++)
   {
      mesh.UniformRefinement();
   }

   // 6. Define a parallel mesh by partitioning the serial mesh.
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();

   // 7. Define a parallel finite element space on the parallel mesh. Here we
   //    use continuous Lagrange finite elements of the specified order.
   //    - If order < 1, we instead use an isoparametric/isogeometric space.
   //    - If the mesh is simplicial and partial assembly is requested,
   //      we use the positive basis, which supports device execution.
   FiniteElementCollection *fec;
   auto basis_type = (pa && pmesh.IsSimplexMesh()) ? BasisType::Positive : BasisType::GaussLobatto;
   if (order > 0)
   {
      fec = new H1_FECollection(order, dim, basis_type);
   }
   else if (pmesh.GetNodes())
   {
      fec = pmesh.GetNodes()->OwnFEC();
      if (myid == 0)
      {
         cout << "Using isoparametric FEs: " << fec->Name() << endl;
      }
   }
   else
   {
      fec = new H1_FECollection(order = 1, dim, basis_type);
   }
   ParFiniteElementSpace fespace(&pmesh, fec);
   HYPRE_BigInt size = fespace.GlobalTrueVSize();
   if (myid == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl;
   }

   // 8. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, the boundary conditions are defined
   //    by marking all the external boundary attributes from the mesh as
   //    essential (Dirichlet) and converting them to a list of true dofs.
   Array<int> ess_tdof_list;
   if (pmesh.bdr_attributes.Size())
   {
      Array<int> ess_bdr(pmesh.bdr_attributes.Max());
      ess_bdr = 0;
      // Apply boundary conditions on all external boundaries:
      pmesh.MarkExternalBoundaries(ess_bdr);
      // Boundary conditions can also be applied based on named attributes:
      // pmesh.MarkNamedBoundaries(set_name, ess_bdr)

      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }

   // 9. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system, which in this case is
   //    (1,phi_i) where phi_i are the basis functions in fespace.
   ParLinearForm b(&fespace);
   ConstantCoefficient one(1.0);
   FunctionCoefficient sine_load(sine_rhs);
   FunctionCoefficient multimode_load(multimode_rhs);
   FunctionCoefficient bubble_exp_load(bubble_exp_rhs);
   Coefficient *rhs = &one;
   if (l2_error)
   {
      rhs = strcmp(mms, "multimode") == 0 ?
            static_cast<Coefficient *>(&multimode_load) :
            strcmp(mms, "bubble-exp") == 0 ?
            static_cast<Coefficient *>(&bubble_exp_load) :
            static_cast<Coefficient *>(&sine_load);
   }
   b.AddDomainIntegrator(new DomainLFIntegrator(*rhs));
   b.Assemble();

   // 10. Define the solution vector x as a parallel finite element grid
   //     function corresponding to fespace. Initialize x with initial guess of
   //     zero, which satisfies the boundary conditions.
   ParGridFunction x(&fespace);
   x = 0.0;

   // 11. Set up the parallel bilinear form a(.,.) on the finite element space
   //     corresponding to the Laplacian operator -Delta, by adding the
   //     Diffusion domain integrator.
   ParBilinearForm a(&fespace);
   if (pa)
   {
      a.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   }
   if (fa)
   {
      a.SetAssemblyLevel(AssemblyLevel::FULL);
      // Sort the matrix column indices when running on GPU or with OpenMP (i.e.
      // when Device::IsEnabled() returns true). This makes the results
      // bit-for-bit deterministic at the cost of somewhat longer run time.
      a.EnableSparseMatrixSorting(Device::IsEnabled());
   }
   a.AddDomainIntegrator(new DiffusionIntegrator(one));

   MPI_Barrier(MPI_COMM_WORLD);
   const double assembly_begin = MPI_Wtime();

   // 12. Assemble the parallel bilinear form and the corresponding linear
   //     system, applying any necessary transformations such as: parallel
   //     assembly, eliminating boundary conditions, applying conforming
   //     constraints for non-conforming AMR, static condensation, etc.
   if (static_cond)
   {
      a.EnableStaticCondensation();
   }
   a.Assemble();

   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
   MPI_Barrier(MPI_COMM_WORLD);
   const double assembly_seconds = MPI_Wtime() - assembly_begin;

   auto print_max_time = [myid](const char *label, double local_time)
   {
      double max_time = 0.0;
      MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0,
                 MPI_COMM_WORLD);
      if (myid == 0)
      {
         cout << label << setprecision(16) << max_time << endl;
      }
   };
   print_max_time("Cold start time: ", cold_start_seconds);
   print_max_time("Operator assembly time: ", assembly_seconds);

   // 13. Solve the linear system A X = B.
   //     * With full assembly, use the BoomerAMG preconditioner from hypre.
   //     * With partial assembly, use Jacobi smoothing or CEED algebraic AMG.
  #ifdef MFEM_USE_CUDSS

   if (!pa && (Device::Allows(Backend::CUDA_MASK) && cudss_solver))
   {
      MPI_Barrier(MPI_COMM_WORLD);
      const double setup_begin = MPI_Wtime();
      CuDSSSolver cudss(MPI_COMM_WORLD);
      cudss.SetMatrixSymType(CuDSSSolver::SYMMETRIC_POSITIVE_DEFINITE);
      cudss.SetMatrixViewType(CuDSSSolver::UPPER);
      cudss.SetOperator(*A);
      print_max_time("Preconditioner setup time: ", MPI_Wtime() - setup_begin);

      MPI_Barrier(MPI_COMM_WORLD);
      const double warmup_begin = MPI_Wtime();
      cudss.Mult(B, X);
      print_max_time("Warm-up solve time: ", MPI_Wtime() - warmup_begin);
      for (int repeat = 0; repeat < profile_repeats; repeat++)
      {
         X = 0.0;
         MPI_Barrier(MPI_COMM_WORLD);
         const double repeat_begin = MPI_Wtime();
         cudss.Mult(B, X);
         print_max_time("Steady-state solve time: ", MPI_Wtime() - repeat_begin);
      }
   }
   else
 #endif
   {
      MPI_Barrier(MPI_COMM_WORLD);
      const double setup_begin = MPI_Wtime();
      Solver *prec = NULL;
      if (pa)
      {
         if (UsesTensorBasis(fespace))
         {
            if (algebraic_ceed)
            {
               // Use algebraic multigrid preconditioner from CEED
               prec = new ceed::AlgebraicSolver(
                         a, ess_tdof_list, ceed_assembled_coarse_solver);
            }
            else
            {
               prec = new OperatorJacobiSmoother(a, ess_tdof_list);
            }
         }
      }
      else
      {
         prec = new HypreBoomerAMG;
      }
      CGSolver cg(MPI_COMM_WORLD);
      cg.SetRelTol(cg_relative_tolerance);
      cg.SetMaxIter(cg_max_iterations);
      cg.SetPrintLevel(profile_repeats > 0 ? -1 : 1);
      if (prec)
      {
         cg.SetPreconditioner(*prec);
      }
      cg.SetOperator(*A);
      print_max_time("Preconditioner setup time: ", MPI_Wtime() - setup_begin);

      MPI_Barrier(MPI_COMM_WORLD);
      const double warmup_begin = MPI_Wtime();
      cg.Mult(B, X);
      print_max_time("Warm-up solve time: ", MPI_Wtime() - warmup_begin);
      if (myid == 0)
      {
         cout << "CG iterations: " << cg.GetNumIterations() << endl;
         cout << "CG converged: " << (cg.GetConverged() ? "yes" : "no") << endl;
         cout << "CG final residual norm: " << setprecision(16)
              << cg.GetFinalNorm() << endl;
      }
      for (int repeat = 0; repeat < profile_repeats; repeat++)
      {
         X = 0.0;
         MPI_Barrier(MPI_COMM_WORLD);
         const double repeat_begin = MPI_Wtime();
         cg.Mult(B, X);
         print_max_time("Steady-state solve time: ", MPI_Wtime() - repeat_begin);
         if (myid == 0)
         {
            cout << "Steady-state CG iterations: " << cg.GetNumIterations() << endl;
            cout << "Steady-state CG converged: "
                 << (cg.GetConverged() ? "yes" : "no") << endl;
            cout << "Steady-state CG final residual norm: " << setprecision(16)
                 << cg.GetFinalNorm() << endl;
         }
      }
      delete prec;
   }


   // 14. Recover the parallel grid function corresponding to X. This is the
   //     local finite element solution on each processor.
   a.RecoverFEMSolution(X, b, x);

   if (l2_error)
   {
      FunctionCoefficient sine_solution(sine_exact_solution);
      FunctionCoefficient multimode_solution(multimode_exact_solution);
      FunctionCoefficient bubble_exp_solution(bubble_exp_exact_solution);
      Coefficient *exact_solution = strcmp(mms, "multimode") == 0 ?
                                    static_cast<Coefficient *>(&multimode_solution) :
                                    strcmp(mms, "bubble-exp") == 0 ?
                                    static_cast<Coefficient *>(&bubble_exp_solution) :
                                    static_cast<Coefficient *>(&sine_solution);
      const real_t error = x.ComputeL2Error(*exact_solution);
      if (myid == 0)
      {
         cout << "L2 norm of error: " << setprecision(16) << error << endl;
      }
   }

   // 15. Save the refined mesh and the solution in parallel. This output can
   //     be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
   if (save_output)
   {
      ostringstream mesh_name, sol_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      sol_name << "sol." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh.Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      x.Save(sol_ofs);
   }

   // 16. Optionally save the solution in ParaView format.
   if (paraview)
   {
      ParaViewDataCollection paraview_dc("Example1P", &pmesh);
      paraview_dc.SetPrefixPath("ParaView");
      if (order > 0)
      {
         paraview_dc.SetLevelsOfDetail(order);
      }
      paraview_dc.SetDataFormat(VTKFormat::BINARY);
      paraview_dc.SetHighOrderOutput(true);
      paraview_dc.SetCycle(0);
      paraview_dc.SetTime(0.0);
      paraview_dc.RegisterField("solution", &x);
      paraview_dc.Save();
   }

   // 17. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int visport = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n"
               << pmesh << x << flush;
   }

   // 18. Free the used memory.
   if (order > 0)
   {
      delete fec;
   }

   return 0;
}

// The manufactured solutions below are written for any dimension: each one is
// a product over the coordinates that vanishes on the whole boundary of the
// unit square (2D) or the unit cube (3D), so it matches the homogeneous
// Dirichlet conditions imposed above. In 2D they reduce to the original 2D
// definitions.

namespace
{

// A separable sine mode, amplitude * prod_i sin(k_i * pi * x_i).
struct SineMode
{
   real_t amplitude;
   int wave_numbers[3];
};

real_t sine_mode_value(const SineMode &mode, const Vector &x)
{
   real_t value = mode.amplitude;
   for (int i = 0; i < x.Size(); i++)
   {
      value *= sin(mode.wave_numbers[i] * M_PI * x[i]);
   }
   return value;
}

// -Delta of a sine mode is pi^2 * (sum_i k_i^2) times the mode itself.
real_t sine_mode_load(const SineMode &mode, const Vector &x)
{
   int wave_numbers_squared = 0;
   for (int i = 0; i < x.Size(); i++)
   {
      wave_numbers_squared += mode.wave_numbers[i] * mode.wave_numbers[i];
   }
   return wave_numbers_squared * M_PI * M_PI * sine_mode_value(mode, x);
}

const SineMode fundamental_mode = {1.0, {1, 1, 1}};

// The trailing wave number is only read in 3D, so in 2D these are the same
// three modes as before.
const SineMode multimode_modes[] =
{
   {1.0, {1, 1, 1}}, {0.1, {3, 2, 1}}, {0.01, {7, 5, 3}}
};

// The bubble g(s) = s * (1 - s) vanishes at s = 0 and s = 1.
real_t bubble(real_t s)
{
   return s * (1.0 - s);
}

} // namespace

real_t sine_exact_solution(const Vector &x)
{
   return sine_mode_value(fundamental_mode, x);
}

real_t sine_rhs(const Vector &x)
{
   return sine_mode_load(fundamental_mode, x);
}

real_t multimode_exact_solution(const Vector &x)
{
   real_t value = 0.0;
   for (const SineMode &mode : multimode_modes)
   {
      value += sine_mode_value(mode, x);
   }
   return value;
}

real_t multimode_rhs(const Vector &x)
{
   real_t value = 0.0;
   for (const SineMode &mode : multimode_modes)
   {
      value += sine_mode_load(mode, x);
   }
   return value;
}

// u = exp(sum_i x_i) * prod_i g(x_i).
real_t bubble_exp_exact_solution(const Vector &x)
{
   real_t exponent = 0.0;
   real_t product = 1.0;
   for (int i = 0; i < x.Size(); i++)
   {
      exponent += x[i];
      product *= bubble(x[i]);
   }
   return exp(exponent) * product;
}

// Differentiating twice picks up (g + 2g' + g'')(s) = -(s^2 + 3s) in each
// direction, so -Delta u is
// exp(sum_i x_i) * sum_i (3*x_i + x_i^2) * prod_{j != i} g(x_j).
real_t bubble_exp_rhs(const Vector &x)
{
   const int dim = x.Size();
   real_t exponent = 0.0;
   for (int i = 0; i < dim; i++)
   {
      exponent += x[i];
   }

   real_t load = 0.0;
   for (int i = 0; i < dim; i++)
   {
      real_t term = 3.0 * x[i] + x[i] * x[i];
      for (int j = 0; j < dim; j++)
      {
         if (j != i)
         {
            term *= bubble(x[j]);
         }
      }
      load += term;
   }
   return exp(exponent) * load;
}
