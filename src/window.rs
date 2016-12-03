use conrod;
use conrod::backend::glium::glium;
use conrod::backend::glium::glium::{DisplayBuild, Surface};
use window_gui;
use std;
use image;
use find_folder;
use img;

const WIN_W: u32 = window_gui::WIN_W;
const WIN_H: u32 = window_gui::WIN_H;

pub fn window_loop() {
    let display = glium::glutin::WindowBuilder::new()
        .with_vsync()
        .with_dimensions(WIN_W, WIN_H)
        .with_title("Mulperc")
        .build_glium()
        .unwrap();

    let mut app = window_gui::MulpercWindow::new();

    let mut ui = conrod::UiBuilder::new([WIN_W as f64, WIN_H as f64]).theme(window_gui::theme()).build();

    let ids = window_gui::Ids::new(ui.widget_id_generator());

    let assets = find_folder::Search::KidsThenParents(3, 5).for_folder("assets").unwrap();
    let font_path = assets.join("fonts/NotoSans/NotoSans-Regular.ttf");
    ui.fonts.insert_from_file(font_path).unwrap();

    use std::path::Path;
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

    fn autoencoder_res(app: &window_gui::MulpercWindow, display: &glium::Display, data: &[f64]) -> glium::texture::Texture2d {
        let empty = image::DynamicImage::ImageRgba8(image::ImageBuffer::from_pixel(1, 1, image::Rgba([0, 0, 0, 0]))).to_rgba();
        match app.net_autoencoder {
            Some(ref encoder) => {
                let res = encoder.feed_forward(data).0;
                let rgba_image = image::DynamicImage::ImageLuma8(image::ImageBuffer::<image::Luma<u8>, _>::from_raw(28, 28, res.at.iter()
                    .map(|f| (f * 255.0) as u8).collect::<Vec<u8>>()).unwrap()).to_rgba();
                let image_dimensions = rgba_image.dimensions();
                let raw_image = glium::texture::RawImage2d::from_raw_rgba_reversed(rgba_image.into_raw(), image_dimensions);
                let texture = glium::texture::Texture2d::new(display, raw_image).unwrap();
                texture
            },
            None => {
                let raw_image = glium::texture::RawImage2d::from_raw_rgba_reversed(empty.into_raw(), (1, 1));
                glium::texture::Texture2d::new(display, raw_image).unwrap()
            }
        }
    }

    let mut old_path: String = assets.join("images/rust.png").to_string_lossy().into();
    let mut old_path_autoencoder: String = assets.join("images/rust.png").to_string_lossy().into();
    let mut old_mnist_idx = 0;

    let mut image_map = window_gui::image_map(&ids,
                                              load_image(&display, &old_path),
                                              mnist(&display, &app.mnist[old_mnist_idx]),
                                              load_image(&display, &old_path_autoencoder),
                                              autoencoder_res(&app, &display, &img::get_pixels(&old_path_autoencoder)));

    let mut renderer = conrod::backend::glium::Renderer::new(&display).unwrap();

    'main: loop {
        for event in display.poll_events() {
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
                        image_map = window_gui::image_map(
                            &ids,
                            load_image(&display, &old_path),
                            mnist(&display, &app.mnist[old_mnist_idx]),
                            load_image(&display, &old_path_autoencoder),
                            autoencoder_res(&app, &display, &img::get_pixels(&old_path_autoencoder))
                        );
                    }
                }

                // Break from the loop upon `Escape`.
                glium::glutin::Event::KeyboardInput(_, _, Some(glium::glutin::VirtualKeyCode::Escape)) |
                glium::glutin::Event::Closed =>
                    break 'main,

                _ => {},
            }
        }

        if let Some(win_rect) = ui.rect_of(ui.window) {
            let (win_w, win_h) = (win_rect.w() as u32, win_rect.h() as u32);
            let (w, h) = display.get_window().unwrap().get_inner_size_points().unwrap();
            if w != win_w || h != win_h {
                let event = conrod::event::Input::Resize(w, h);
                ui.handle_event(event);
            }
        }

        if ui.global_input.events().next().is_some() {
            let mut ui = ui.set_widgets();
            window_gui::gui(&mut ui, &ids, &mut app);
        }

        if app.image_path != old_path && app.image_path != "" {
            use img;
            old_path = app.image_path.clone();
            app.image = Some(img::get_pixels(&old_path));
            image_map = window_gui::image_map(
                &ids,
                load_image(&display, &old_path),
                mnist(&display, &app.mnist[old_mnist_idx]),
                load_image(&display, &old_path_autoencoder),
                autoencoder_res(&app, &display, &img::get_pixels(&old_path_autoencoder))
            );
        }

        if app.mnist_idx != old_mnist_idx {
            old_mnist_idx = app.mnist_idx;
            image_map = window_gui::image_map(
                &ids,
                load_image(&display, &old_path),
                mnist(&display, &app.mnist[old_mnist_idx]),
                load_image(&display, &old_path_autoencoder),
                autoencoder_res(&app, &display, &img::get_pixels(&old_path_autoencoder))
            );
        }

        if let Some(primitives) = ui.draw_if_changed() {
            renderer.fill(&display, primitives, &image_map);
            let mut target = display.draw();
            target.clear_color(0.0, 0.0, 0.0, 1.0);
            renderer.draw(&display, &mut target, &image_map).unwrap();
            target.finish().unwrap();
        }

        std::thread::sleep(std::time::Duration::from_millis(16));
    }
}