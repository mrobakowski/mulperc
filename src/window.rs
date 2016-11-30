use conrod;
use conrod::backend::glium::glium;
use conrod::backend::glium::glium::{DisplayBuild, Surface};
use window_gui;
use std;
use image;
use find_folder;

// The initial width and height in "points".
const WIN_W: u32 = window_gui::WIN_W;
const WIN_H: u32 = window_gui::WIN_H;

pub fn window_loop() {
    // Build the window.
    let display = glium::glutin::WindowBuilder::new()
        .with_vsync()
        .with_dimensions(WIN_W, WIN_H)
        .with_title("Mulperc")
        .build_glium()
        .unwrap();

    // A demonstration of some app state that we want to control with the conrod GUI.
    let mut app = window_gui::MulpercWindow::new();

    // Construct our `Ui`.
    let mut ui = conrod::UiBuilder::new([WIN_W as f64, WIN_H as f64]).theme(window_gui::theme()).build();

    // The `widget::Id` of each widget instantiated in `support::gui`.
    let ids = window_gui::Ids::new(ui.widget_id_generator());

    // Add a `Font` to the `Ui`'s `font::Map` from file.
    let assets = find_folder::Search::KidsThenParents(3, 5).for_folder("assets").unwrap();
    let font_path = assets.join("fonts/NotoSans/NotoSans-Regular.ttf");
    ui.fonts.insert_from_file(font_path).unwrap();

    use std::path::Path;
    // Load the Rust logo from our assets folder to use as an example image.
    fn load_image<P: AsRef<Path>>(display: &glium::Display, path: P) -> glium::texture::Texture2d {
        let rgba_image = image::open(path.as_ref()).unwrap().to_rgba();
        let image_dimensions = rgba_image.dimensions();
        let raw_image = glium::texture::RawImage2d::from_raw_rgba_reversed(rgba_image.into_raw(), image_dimensions);
        let texture = glium::texture::Texture2d::new(display, raw_image).unwrap();
        texture
    }

    fn mnist(display: &glium::Display, datapoint: &(Vec<f64>, String)) -> glium::texture::Texture2d {
        let rgba_image = image::DynamicImage::ImageLuma8(image::ImageBuffer::<image::Luma<u8>, _>::from_raw(28, 28, datapoint.0.iter()
            .map(|f| (f * 255.0) as u8).collect::<Vec<u8>>()).unwrap()).to_rgba();
        let image_dimensions = rgba_image.dimensions();
        let raw_image = glium::texture::RawImage2d::from_raw_rgba_reversed(rgba_image.into_raw(), image_dimensions);
        let texture = glium::texture::Texture2d::new(display, raw_image).unwrap();
        texture
    }

    let mut old_path: String = assets.join("images/rust.png").to_string_lossy().into();
    let mut old_mnist_idx = 0;

    let mut image_map = window_gui::image_map(&ids, load_image(&display, &old_path), mnist(&display, &app.mnist[old_mnist_idx]));

    // A type used for converting `conrod::render::Primitives` into `Command`s that can be used
    // for drawing to the glium `Surface`.
    //
    // Internally, the `Renderer` maintains:
    // - a `backend::glium::GlyphCache` for caching text onto a `glium::texture::Texture2d`.
    // - a `glium::Program` to use as the shader program when drawing to the `glium::Surface`.
    // - a `Vec` for collecting `backend::glium::Vertex`s generated when translating the
    // `conrod::render::Primitive`s.
    // - a `Vec` of commands that describe how to draw the vertices.
    let mut renderer = conrod::backend::glium::Renderer::new(&display).unwrap();

    // Start the loop:
    //
    // - Render the current state of the `Ui`.
    // - Update the widgets via the `support::gui` fn.
    // - Poll the window for available events.
    // - Repeat.
    'main: loop {
        // Poll for events.
        for event in display.poll_events() {
            // Use the `glutin` backend feature to convert the glutin event to a conrod one.
            let window = display.get_window().unwrap();
            if let Some(event) = conrod::backend::glutin::convert(event.clone(), window) {
                ui.handle_event(event);
            }

            match event {
                glium::glutin::Event::DroppedFile(path) => {
                    let path = path.to_string_lossy();

                    if path != old_path && path != "" {
                        use img;
                        app.image_path = path.into();
                        old_path = app.image_path.clone();
                        app.image = Some(img::get_pixels(&old_path));
                        image_map = window_gui::image_map(&ids, load_image(&display, &old_path), mnist(&display, &app.mnist[old_mnist_idx]));
                    }
                }

                // Break from the loop upon `Escape`.
                glium::glutin::Event::KeyboardInput(_, _, Some(glium::glutin::VirtualKeyCode::Escape)) |
                glium::glutin::Event::Closed =>
                    break 'main,

                _ => {},
            }
        }

        // We must manually track the window width and height as it is currently not possible to
        // receive `Resize` events from glium on Mac OS any other way.
        //
        // TODO: Once the following PR lands, we should stop tracking size like this and use the
        // `window_resize_callback`. https://github.com/tomaka/winit/pull/88
        if let Some(win_rect) = ui.rect_of(ui.window) {
            let (win_w, win_h) = (win_rect.w() as u32, win_rect.h() as u32);
            let (w, h) = display.get_window().unwrap().get_inner_size_points().unwrap();
            if w != win_w || h != win_h {
                let event = conrod::event::Input::Resize(w, h);
                ui.handle_event(event);
            }
        }

        // If some input event has been received, update the GUI.
        if ui.global_input.events().next().is_some() {
            // Instantiate a GUI demonstrating every widget type provided by conrod.
            let mut ui = ui.set_widgets();
            window_gui::gui(&mut ui, &ids, &mut app);
        }

        if app.image_path != old_path && app.image_path != "" {
            use img;
            old_path = app.image_path.clone();
            app.image = Some(img::get_pixels(&old_path));
            image_map = window_gui::image_map(&ids, load_image(&display, &old_path), mnist(&display, &app.mnist[old_mnist_idx]));
        }

        if app.mnist_idx != old_mnist_idx {
            old_mnist_idx = app.mnist_idx;
            image_map = window_gui::image_map(&ids, load_image(&display, &old_path), mnist(&display, &app.mnist[old_mnist_idx]));
        }

        // Draw the `Ui`.
        if let Some(primitives) = ui.draw_if_changed() {
            renderer.fill(&display, primitives, &image_map);
            let mut target = display.draw();
            target.clear_color(0.0, 0.0, 0.0, 1.0);
            renderer.draw(&display, &mut target, &image_map).unwrap();
            target.finish().unwrap();
        }

        // Avoid hogging the CPU.
        std::thread::sleep(std::time::Duration::from_millis(16));
    }
}